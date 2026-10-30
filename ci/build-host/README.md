# The Linux build host

A self-hosted machine that runs `pr.yml`'s Linux jobs for PRs from branches of this repository, so they don't wait for hosted runners ([ADR-0625](../../plan/decisions/ADR-0625-linux-build-host.md)). Fork PRs, merge groups and every other run stay on hosted runners. The host is not a lab rig and never runs a lab job.

| File | What it does |
|---|---|
| `Dockerfile` | The job image: Ubuntu 24.04, the pinned runner release, a C toolchain, Python 3, sudo, and the toolchain `rust-toolchain.toml` pins |
| `runner-slot.sh` | One slot run: mints a single-use just-in-time runner configuration labelled `syntropy-build`, then runs one job in a fresh container |
| `syntropy-build-runner@.service` | The systemd unit for one slot; it restarts the slot after each job |
| `egress.sh` | The job network: containers reach the internet, but no private, link-local or tailnet address and not the host |
| `install.sh` | Builds the image, installs the scripts and the unit, and starts the slots |

## How a job reaches it

`pr.yml` names the label only in the gated form the workflow lint checks:

```yaml
runs-on: ${{ (vars.SYNTROPY_BUILD_HOST == 'on' && github.event.pull_request.head.repo.full_name == github.repository) && 'syntropy-build' || 'ubuntu-24.04' }}
```

While the repository variable `SYNTROPY_BUILD_HOST` is not `on`, every job runs on hosted runners.

## Setting it up (Maintainer action)

The machine needs x86-64 Linux, Docker, systemd, `jq` and `curl`.

1. Create a fine-grained personal access token. Set the resource owner to `syntropyai-hq`, select only the `syntropy` repository, and grant the repository permission **Administration: Read and write**, the permission GitHub requires to register repository runners. Give it an expiry of 90 days or less.
2. On the host, as root:
   ```sh
   install -d -m 0700 /etc/syntropy-build
   ( umask 077; cat > /etc/syntropy-build/token )   # paste the token, then Ctrl-D
   ```
3. From a checkout of `main` on the host: `sudo ci/build-host/install.sh 3` (3 slots).
4. Check that the runners are listed as idle: `gh api repos/syntropyai-hq/syntropy/actions/runners --jq '.runners[] | [.name, .status, .busy] | @tsv'`.
5. Turn the gate on: `gh variable set SYNTROPY_BUILD_HOST --body on --repo syntropyai-hq/syntropy`.

The machine must not suspend while the gate is on. On Ubuntu: `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`.

## Running it

- **Taking the host down.** First run `gh variable set SYNTROPY_BUILD_HOST --body off --repo syntropyai-hq/syntropy`. Jobs queued for `syntropy-build` before the change still wait for the host, so re-run them.
- **Updating the image** after a change to `rust-toolchain.toml`, the runner release or this directory: pull `main` and run `install.sh` again. It restarts every slot, and a restart stops a running job after 30 seconds, so run it when the runners are idle.
- **Rotating the token**: replace `/etc/syntropy-build/token` and restart the slots: `systemctl restart 'syntropy-build-runner@*'`.
- **Logs**: `journalctl -u 'syntropy-build-runner@*' -f`.

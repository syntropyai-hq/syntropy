# The Linux build host

A self-hosted machine that runs the workflows' Linux jobs, so they don't wait for hosted runners ([ADR-0625](../../plan/decisions/ADR-0625-linux-build-host.md)): PRs from branches of this repository, merge groups, pushes, schedules and manual runs. Fork PRs, the macOS and Windows jobs, `toolchains-linux` (it needs `docker buildx`) and `dep-approval.yml` (it waits for build-host jobs, so it must not hold a slot) stay on hosted runners. A merge group's run includes the code of every queued PR, a fork's among them once a maintainer has queued it. The host is not a lab rig and never runs a lab job.

| File | What it does |
|---|---|
| `Dockerfile` | The job image: Ubuntu 24.04, the pinned runner release, a C toolchain, Python 3, sudo, and the toolchain `rust-toolchain.toml` pins |
| `job-started.sh` | The runner's job-started hook, baked into the image: it fails any job that isn't one of this repository's workflows, or that carries a fork's PR, before the job's first step |
| `runner-slot.sh` | One slot run: mints a single-use just-in-time runner configuration labelled `syntropy-build`, then runs one job in a fresh container |
| `syntropy-build-runner@.service` | The systemd unit for one slot; it restarts the slot after each job |
| `egress.sh` | The job network: containers reach the internet, but no private, link-local or tailnet address and not the host |
| `install.sh` | Builds the image, installs the scripts and the units, starts the slots and enables the daily update |
| `update.sh`, `syntropy-build-update.{service,timer}` | Daily: fetches `main` and, when this directory or `rust-toolchain.toml` changed, rebuilds the image, reinstalls, and restarts the idle slots |
| `watchdog.sh` | Run by `build-host-watchdog.yml` on a hosted runner every 10 minutes: sets the gate to `auto-off` while no build-host runner is online, re-runs the runs waiting for the host, and sets it back to `on` when one is |

## How a job reaches it

A workflow names the label only in the gated form the workflow lint checks:

```yaml
runs-on: ${{ (vars.SYNTROPY_BUILD_HOST == 'on' && (!github.event.pull_request || github.event.pull_request.head.repo.full_name == github.repository)) && 'syntropy-build' || 'ubuntu-24.04' }}
```

While the repository variable `SYNTROPY_BUILD_HOST` is not `on`, every job runs on hosted runners. The watchdog moves it between `on` and `auto-off` by itself; `off`, or any other value, is a maintainer's and the watchdog leaves it alone.

## What protects the host

- **No fork PRs.** The gate sends every event but a fork's PR, the workflow lint refuses the label anywhere else, and the job-started hook refuses any other job before its first step, even one a maintainer approved by mistake.
- **One job per container.** Each job gets a fresh container and a single-use registration, and the container is removed afterwards, so nothing a job does persists.
- **No credential in a job.** The app key or token stays on the host, readable only by root, and reaches `curl` on stdin. A container gets only its own single-use configuration.
- **No LAN.** Job containers reach the internet and nothing else: no private, link-local or tailnet address, and not the host.
- **Least privilege.** Containers drop every capability but the few the jobs need, with no `NET_RAW`. The slot unit runs with only the capabilities iptables needs, behind a system-call filter (`systemd-analyze security` rates it 2.0).

## Setting it up (Maintainer action)

The machine needs x86-64 Linux, Docker, systemd, git, `jq`, `curl` and `openssl`.

1. **The runner app.** In the organisation's settings, **Developer settings → GitHub Apps → New GitHub App**: name it `syntropy-build-host`, untick **Webhook → Active**, and grant only the repository permission **Administration: Read and write**. Create it, note its **App ID**, generate a **private key** (a `.pem` file), and install the app on the `syntropy` repository only.
2. On the host, as root, put the key and the ID in `/etc/syntropy-build`:
   ```sh
   install -d -m 0700 /etc/syntropy-build
   install -m 0600 syntropy-build-host.*.private-key.pem /etc/syntropy-build/app-key.pem
   ( umask 077; echo <App ID> > /etc/syntropy-build/app-id )
   ```
   Then delete the downloaded `.pem`. Without an app, a fine-grained token works instead: resource owner `syntropyai-hq`, only the `syntropy` repository, **Administration: Read and write**, an expiry of 90 days or less, in `/etc/syntropy-build/token`.
3. From a checkout of `main` on the host: `sudo ci/build-host/install.sh 3` (3 slots).
4. Check that the runners are listed as idle: `gh api repos/syntropyai-hq/syntropy/actions/runners --jq '.runners[] | [.name, .status, .busy] | @tsv'`.
5. Turn the gate on: `gh variable set SYNTROPY_BUILD_HOST --body on --repo syntropyai-hq/syntropy`.
6. **The watchdog app.** Create a second GitHub App the same way, named `syntropy-build-watchdog`, with only **Administration: Read-only** and **Variables: Read and write**, and install it on the `syntropy` repository only. Store its private key as the repository secret `BUILD_HOST_WATCHDOG_KEY` and its App ID as the repository variable `BUILD_HOST_WATCHDOG_APP_ID`. The watchdog runs from the next 10-minute mark.

The machine must not suspend while the gate is on. On Ubuntu: `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`.

## Running it

- **Taking the host down.** First run `gh variable set SYNTROPY_BUILD_HOST --body off --repo syntropyai-hq/syntropy`. Jobs queued for `syntropy-build` before the change still wait for the host, so re-run them.
- **Updates** happen by themselves: each day `syntropy-build-update.timer` brings the host up to date with `main`. To update now: `sudo systemctl start syntropy-build-update`. `install.sh` restarts every slot, and a restart stops a running job after 30 seconds, so run it only when the runners are idle.
- **Changing the credential**: replace the files in `/etc/syntropy-build`. Each slot reads them at its next run, after its current job; no restart is needed. When `app-key.pem` is there the token is not used.
- **The gate**: `auto-off` means the watchdog found no runner online. To keep the host off whatever the watchdog sees, set it to `off`.
- **Logs**: `journalctl -u 'syntropy-build-runner@*' -f`.

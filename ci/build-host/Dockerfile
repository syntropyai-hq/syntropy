# SPDX-License-Identifier: MIT OR Apache-2.0

# The job image of the Linux build host (ADR-0625). Each job runs in a fresh container of this
# image, as an ephemeral runner from a single-use just-in-time configuration, and the container is
# removed when the job ends. The image has what `pr.yml`'s Linux jobs expect from `ubuntu-24.04`:
# git, a C toolchain, Python 3, curl, sudo and rustup. The toolchain `rust-toolchain.toml` pins is
# installed here, so a job's `setup-rust` step finds it and downloads nothing.
#
# Build it from the repository root, so `rust-toolchain.toml` is in the context:
#   docker build -f ci/build-host/Dockerfile -t syntropy-build:latest .

FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

# The runner release and its SHA-256, from the release notes of actions/runner. A runner older
# than GitHub's supported window updates itself before its job, which costs time but not
# correctness; bump both together.
ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
ARG RUSTUP_VERSION=1.29.2
ARG RUSTUP_SHA256=5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git jq pkg-config python3 sudo tar unzip xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*

# The job user. Passwordless sudo, as on a hosted runner: `hosted-test` adds a second OS account
# for the IPC peer test. The container is unprivileged and removed after the job, so sudo reaches
# only the container.
RUN useradd --create-home --uid 1001 --shell /bin/bash runner \
 && echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner \
 && chmod 0440 /etc/sudoers.d/runner

WORKDIR /home/runner
RUN curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 --retry 3 \
      --output runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
 && echo "${RUNNER_SHA256}  runner.tar.gz" | sha256sum --check --strict \
 && tar -xzf runner.tar.gz && rm runner.tar.gz \
 && ./bin/installdependencies.sh \
 && rm -rf /var/lib/apt/lists/* \
 && chown -R runner:runner /home/runner

USER runner
ENV PATH=/home/runner/.cargo/bin:$PATH
COPY --chown=runner:runner rust-toolchain.toml /tmp/toolchain/rust-toolchain.toml
RUN curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 --retry 3 \
      --output /tmp/rustup-init \
      "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
 && echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum --check --strict \
 && chmod +x /tmp/rustup-init \
 && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain none \
 && rm /tmp/rustup-init \
 && cd /tmp/toolchain && rustup toolchain install --no-self-update \
 && rustup default "$(rustup show active-toolchain | cut -d ' ' -f 1)"

# The supervisor passes the configuration in `JITCONFIG`. It is single-use: once this runner has
# registered, the configuration registers nothing else.
ENTRYPOINT ["/bin/bash", "-c", "exec ./run.sh --jitconfig \"$JITCONFIG\""]

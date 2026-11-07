# syntax=docker/dockerfile:1.27.1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The Linux build image (RFC-0062 §4.1, §4.2; QUAL-E16-T24): Ubuntu 22.04, so the binaries it
# builds need nothing newer than glibc 2.35 (ADR-0020's floor), with the CUDA 13 toolkit, CMake
# and a C++ compiler for the vendored ggml and the statically built HiGHS (ADR-0008).
#
# Every input is pinned, so a rebuild from the same file gives the same image, byte for byte:
# - the base image and this file's syntax frontend, by digest;
# - the Ubuntu packages, by version, from one Ubuntu archive snapshot;
# - the CUDA components and CMake, by the SHA-256 their publishers list, checked as they are
#   fetched;
# - every timestamp: build with `SOURCE_DATE_EPOCH` set and the exporter's `rewrite-timestamp`,
#   and the files whose contents record the time of the build or the machine are removed.
#
# `ci/toolchains.toml` repeats each pin, and `lint`'s `toolchains` step checks that the two
# agree. `pr.yml`'s `toolchains-linux` job builds this image twice and compares the digests, then
# prints each tool's version inside it. To build it the same way locally:
#
#   docker buildx build --platform linux/amd64 --no-cache \
#     --build-arg SOURCE_DATE_EPOCH=1792454400 \
#     --output type=docker,dest=cuda-13.tar,rewrite-timestamp=true ci/images/cuda-13

ARG BASE=ubuntu:jammy-20260930@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956

# --- The Ubuntu archive, from one snapshot. ---
FROM ${BASE} AS apt

# The snapshot's packages are verified by the archive's signed Release file, with the keyring in
# the base image, so the transport needs no certificate store, which the base lacks. The snapshot
# service serves only HTTPS, so peer verification is off for it alone.
ARG APT_SNAPSHOT=20261020T000000Z
RUN set -eu; \
    url="https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}"; \
    printf '%s\n' \
        "deb ${url} jammy main" \
        "deb ${url} jammy-updates main" \
        "deb ${url} jammy-security main" \
        > /etc/apt/sources.list; \
    printf '%s\n' \
        'Acquire::https::snapshot.ubuntu.com::Verify-Peer "false";' \
        'Acquire::Check-Valid-Until "false";' \
        'APT::Install-Recommends "false";' \
        > /etc/apt/apt.conf.d/90syntropy

# --- Downloads, unpacked into /opt; only the trees are copied into the image. ---
FROM apt AS fetch

RUN set -eu; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xz-utils=5.2.5-2ubuntu1.1

# The CUDA 13.4.3 components a kernel build needs: the compiler driver, its front end and NVVM,
# the runtime and CCCL. The digests are those of NVIDIA's redistrib_13.4.3.json.
ADD --checksum=sha256:60998f40cc9df5826b2a846e2dcc328133fbc5ef539398ce0a21cf5dcac7193d https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.4.92-archive.tar.xz /fetch/cuda/
ADD --checksum=sha256:c969e61ded12dbf0cf20ace72867f1622ec0e8ea1e1d305e27761012025f7aaa https://developer.download.nvidia.com/compute/cuda/redist/cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.4.92-archive.tar.xz /fetch/cuda/
ADD --checksum=sha256:0e619cf3d5b27f7c812665599ca9761fc296e56129b24a85d822a5d8a1a3bd75 https://developer.download.nvidia.com/compute/cuda/redist/libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.4.92-archive.tar.xz /fetch/cuda/
ADD --checksum=sha256:0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5 https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.4.92-archive.tar.xz /fetch/cuda/
ADD --checksum=sha256:6b7516074f42f80dd3feb9c11318c8e1c5ab56cdd9f18fff4c5ace2e7e0a8209 https://developer.download.nvidia.com/compute/cuda/redist/cccl/linux-x86_64/cccl-linux-x86_64-13.3.4.3.1-archive.tar.xz /fetch/cuda/
# CMake 4.4.4, with the digest of Kitware's cmake-4.4.4-SHA-256.txt.
ADD --checksum=sha256:e5bb807f7728cb60cd8b27ebc97a2edb469b68655f21e844a600c3575b76f5bb https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-linux-x86_64.tar.gz /fetch/cmake/

# Each archive holds one top directory; its contents are merged into one tree per tool.
RUN set -eu; \
    mkdir -p /opt/cuda /opt/cmake; \
    for archive in /fetch/cuda/*.tar.xz; do \
        tar -xJf "$archive" -C /opt/cuda --strip-components=1 --no-same-owner; \
    done; \
    tar -xzf /fetch/cmake/cmake-4.4.4-linux-x86_64.tar.gz -C /opt/cmake --strip-components=1 \
        --no-same-owner; \
    chmod -R u=rwX,go=rX /opt/cuda /opt/cmake

# --- The image. ---
FROM apt

# The time every file and the image config carry: the snapshot's own time, so a rebuild from the
# same pins needs no other input.
ARG SOURCE_DATE_EPOCH=1792454400

# After the install, the files that record the time of the build go, with the caches the next
# apt run rebuilds.
RUN set -eu; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++-11=11.4.0-1ubuntu1~22.04.3 \
        g++=4:11.2.0-1ubuntu1 \
        make=4.3-4.1build1; \
    rm /etc/apt/apt.conf.d/90syntropy; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/cache/debconf/*-old \
        /var/cache/ldconfig/aux-cache /var/lib/dpkg/*-old /var/log/*

COPY --from=fetch /opt/cuda /usr/local/cuda-13.4
COPY --from=fetch /opt/cmake /opt/cmake

ENV CUDA_PATH=/usr/local/cuda-13.4
ENV PATH=/usr/local/cuda-13.4/bin:/opt/cmake/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

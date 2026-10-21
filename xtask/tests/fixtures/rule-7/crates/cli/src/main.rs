// SPDX-License-Identifier: MIT OR Apache-2.0

//! The core binary of this fixture, which links a provider and so a vendor GPU library.

#![forbid(unsafe_code)]

fn main() {
    std::process::exit(syntropy_kernels_cuda::init());
}

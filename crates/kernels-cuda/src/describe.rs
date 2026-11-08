// SPDX-License-Identifier: MIT OR Apache-2.0

//! What enumeration decides from the driver's facts: the driver check, the device class and the
//! feature bits (RFC-0040 §2.2, §7.1; ADR-0014). Pure functions, so the tests cover every driver
//! version and compute capability without a GPU.

use syntropy_core::resource::{DeviceKind, PciAddress};
use syntropy_core::{Bytes, ErrorClass};
use syntropy_kernels::error::codes;
use syntropy_kernels::{DeviceClass, DeviceFeatures, DeviceMemoryInfo, Error, ProviderDevice};

use crate::driver::RawDevice;

/// The oldest driver the provider runs on: CUDA 13.0's driver API, which driver branch 580 first
/// ships (ADR-0014; RFC-0040 §7.1). `cuDriverGetVersion` reports 1000 × major + 10 × minor.
pub const MIN_DRIVER_API: i32 = 13_000;

/// The oldest architecture CUDA 13 builds for: `sm_75`, Turing (ADR-0014).
pub const MIN_SM: u16 = 75;

/// A driver version as people read it, e.g. `13.5` for 13050.
#[must_use]
pub fn driver_text(version: i32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

/// `Ok` when the driver is new enough, else `DriverTooOld`, `(Kernels, 5)`, class
/// `Unsupported`: the backend is skipped and the next domain is tried (RFC-0040 §13, §6.4).
pub fn check_driver(version: i32) -> Result<(), Error> {
    if version >= MIN_DRIVER_API {
        Ok(())
    } else {
        Err(Error::new(
            ErrorClass::Unsupported,
            codes::DRIVER_TOO_OLD,
            format!(
                "the CUDA driver supports CUDA {}; the provider needs CUDA {} (driver 580 or newer)",
                driver_text(version),
                driver_text(MIN_DRIVER_API)
            ),
        ))
    }
}

/// The `sm` number of a compute capability, e.g. 89 for 8.9; `None` when it does not fit.
#[must_use]
pub fn sm(cc: (i32, i32)) -> Option<u16> {
    u16::try_from(cc.0.checked_mul(10)?.checked_add(cc.1)?).ok()
}

/// The feature bits a device has (ADR-0014 rule 1). The matrix-unit bits follow from the compute
/// capability; the memory bits are the driver's device attributes. `STREAM_MEM_OPS` is never set
/// here: COMP-E4-T14's functional probe decides it, and no attribute is read for it.
#[must_use]
pub fn features(raw: &RawDevice, sm: u16) -> DeviceFeatures {
    let mut f = DeviceFeatures::empty();
    // dp4a from sm_61, FP16 MMA from sm_70 and INT8 MMA from sm_75: every device this provider
    // claims has them.
    f |= DeviceFeatures::INT_DOT | DeviceFeatures::FP16_MMA | DeviceFeatures::INT8_TC;
    if sm >= 80 {
        f |= DeviceFeatures::BF16_MMA;
    }
    if sm >= 89 {
        f |= DeviceFeatures::FP8_TC;
    }
    if sm >= 100 {
        f |= DeviceFeatures::FP4_TC;
    }
    // Graph capture and `cuMemcpyBatchAsync` come with every CUDA 13 driver.
    f |= DeviceFeatures::GRAPH_CAPTURE | DeviceFeatures::BATCHED_COPY;
    if raw.vmm {
        f |= DeviceFeatures::VMM;
    }
    if raw.host_register {
        f |= DeviceFeatures::HOST_REGISTER;
    }
    if raw.peer {
        f |= DeviceFeatures::P2P;
    }
    if raw.integrated {
        f |= DeviceFeatures::UNIFIED;
    }
    f
}

/// The record for one device, or `None` for a device older than `sm_75`, which no CUDA 13 kernel
/// runs on (ADR-0014). `local` is the provider-local index.
#[must_use]
pub fn describe(local: u32, raw: &RawDevice) -> Option<ProviderDevice> {
    let sm = sm(raw.cc).filter(|&sm| sm >= MIN_SM)?;
    let pci = PciAddress {
        domain: raw.pci.0,
        bus: u8::try_from(raw.pci.1).ok()?,
        device: u8::try_from(raw.pci.2).ok()?,
        function: 0,
    };
    Some(ProviderDevice {
        local,
        uuid: Some(raw.uuid),
        pci: Some(pci),
        kind: if raw.integrated {
            DeviceKind::IntegratedGpu
        } else {
            DeviceKind::DiscreteGpu
        },
        name: raw.name.clone(),
        class: DeviceClass::Cuda { sm },
        features: features(raw, sm),
        memory: DeviceMemoryInfo {
            total: Bytes(raw.total),
            free: Bytes(raw.free),
            budget: None,
            host_visible_device_local: None,
            max_alloc: Bytes(raw.total),
            unified: raw.integrated,
        },
        numa: raw.numa.and_then(|n| u16::try_from(n).ok()),
    })
}

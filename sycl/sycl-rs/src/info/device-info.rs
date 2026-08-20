//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs_sys::device::ffi;

use crate::{
    device::{Aspect, Device},
    info::Info,
    private::Sealed,
};

/// Returns the device type associated with the device. May not return `sycl_rs::info::DeviceType::All`
pub struct DeviceType;
impl Sealed for DeviceType {}
impl Info for DeviceType {
    type Item = crate::info::DeviceType;
    type Target = Device;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_device_type(&target.0)
    }
}

/// Returns a backend-defined device version.
pub struct Version;
impl Sealed for Version {}
impl Info for Version {
    type Item = String;
    type Target = Device;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_version(&target.0)
    }
}

/// Returns the device name of this SYCL device.
pub struct Name;
impl Sealed for Name {}
impl Info for Name {
    type Item = String;
    type Target = Device;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_name(&target.0)
    }
}

/// Returns the PCI BDF address reported by the Intel SYCL extension.
/// Panics when the device does not support the extension.
pub struct PciBdfAddress;
impl Sealed for PciBdfAddress {}
impl Info for PciBdfAddress {
    type Item = String;
    type Target = Device;
    fn get_item(target: &Self::Target) -> Self::Item {
        assert!(
            target.has(Aspect::ExtIntelPciAddress),
            "device does not support the Intel PCI address extension"
        );
        ffi::get_pci_bdf_address(&target.0)
    }
}

//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs_sys::device::ffi;

use crate::{info::InfoTarget, platform::Platform};

/// The `Device` struct encapsulates a single SYCL device on which kernels can be executed.
///
/// The `Device` struct provides the common reference semantics.
pub struct Device(pub(crate) cxx::UniquePtr<ffi::Device>);

impl InfoTarget for Device {}

impl From<cxx::UniquePtr<ffi::Device>> for Device {
    fn from(value: cxx::UniquePtr<ffi::Device>) -> Self {
        Self(value)
    }
}

impl Device {
    /// Returns a [`Vec`] containing all the root devices from all SYCL backends
    /// available in the system which have the device type encapsulated by [`DeviceType`](crate::info::DeviceType).
    pub fn get_devices() -> Vec<Self> {
        ffi::get_devices()
            .into_iter()
            .map(|device| Self(device.ptr))
            .collect()
    }

    /// Returns the associated SYCL platform.
    pub fn get_platform(&self) -> Platform {
        let raw_platform = ffi::get_platform(&self.0);
        Platform(raw_platform)
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        ffi::clone(&self.0).into()
    }
}

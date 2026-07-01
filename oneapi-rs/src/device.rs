//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs_sys::device::ffi;

use crate::{info::device::DeviceInfo, platform::Platform};

pub struct Device(pub(crate) cxx::UniquePtr<ffi::Device>);

impl Device {
    pub fn get_devices() -> Vec<Self> {
        ffi::Device::get_devices()
            .into_iter()
            .map(|device| Self(device.ptr))
            .collect()
    }

    pub fn get_info<T: DeviceInfo>(&self) -> T::Item {
        T::get_item(self)
    }

    pub fn get_platform(&self) -> Platform {
        let raw_platform = self.0.get_platform();
        Platform(raw_platform)
    }
}

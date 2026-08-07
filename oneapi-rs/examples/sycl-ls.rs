//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs::prelude::*;

fn main() {
    for platform in Platform::all() {
        for device in platform.devices() {
            let device_type = device.info::<info::device::DeviceType>();
            let platform_name = platform.info::<info::platform::Name>();
            let device_name = device.info::<info::device::Name>();
            let device_version = device.info::<info::device::Version>();
            let platform_version = platform.info::<info::platform::Version>();

            println!(
                "[{device_type:?}] {platform_name}, {device_name} {device_version} [{platform_version}]"
            );
        }
    }
}

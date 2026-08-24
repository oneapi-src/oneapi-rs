//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs::prelude::*;

fn main() {
    for platform in Platform::get_platforms() {
        for device in platform.get_devices() {
            let device_type = device.get_info::<info::device::DeviceType>();
            let platform_name = platform.get_info::<info::platform::Name>();
            let device_name = device.get_info::<info::device::Name>();
            let device_version = device.get_info::<info::device::Version>();
            let platform_version = platform.get_info::<info::platform::Version>();

            let pci_bdf_address = if device.has(Aspect::ExtIntelPciAddress) {
                device.get_info::<info::device::PciBdfAddress>()
            } else {
                String::from("N/A")
            };

            let integrated_status = if device.has(Aspect::ExtOneapiIsIntegratedGpu) {
                "Integrated "
            } else {
                ""
            };

            println!(
                "[{integrated_status}{device_type:?}] {platform_name}, {device_name} \
                {device_version} PCI:{pci_bdf_address} [{platform_version}]"
            );
        }
    }
}

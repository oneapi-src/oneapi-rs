//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::device")]
pub mod ffi {
    #[namespace = "sycl_shims"]
    extern "C++" {
        include!("sycl-rs-sys/src/types-sys.rs.h");
        type DevicePtr = crate::types::ffi::DevicePtr;
        type PlatformPtr = crate::types::ffi::PlatformPtr;
        type DeviceType = crate::types::ffi::DeviceType;
        type Aspect = crate::types::ffi::Aspect;
    }

    unsafe extern "C++" {
        include!("sycl-rs-sys/include/device.hpp");

        #[namespace = "sycl_shims"]
        type Device = crate::types::ffi::Device;
        #[namespace = "sycl_shims"]
        type Platform = crate::types::ffi::Platform;

        fn get_devices() -> Vec<DevicePtr>;
        fn get_platform(device: &Device) -> UniquePtr<Platform>;
        fn get_device_type(device: &Device) -> DeviceType;
        fn get_version(device: &Device) -> String;
        fn get_name(device: &Device) -> String;
        fn get_pci_bdf_address(device: &Device) -> String;

        fn clone(device: &Device) -> UniquePtr<Device>;
        fn has(device: &Device, aspect: Aspect) -> bool;
    }
}

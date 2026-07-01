//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims")]
pub mod ffi {
    // This is a workaround - cxx currently doesn't support passing
    // around vectors of pointers directly
    // https://github.com/dtolnay/cxx/issues/774#issuecomment-808674945
    struct DevicePtr {
        ptr: UniquePtr<Device>
    }

    #[derive(Debug)]
    enum DeviceType {
        Cpu,
        Gpu,
        Accelerator,
        Custom,
        Automatic,
        All,
        Unimplemented
    }

    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/device.hpp");
 
        type Device;
        type Platform = crate::platform::ffi::Platform;

        #[Self = "Device"]
        fn get_devices() -> Vec<DevicePtr>;

        fn get_platform(self: &Device) -> UniquePtr<Platform>;

        fn get_device_type(self: &Device) -> DeviceType;
        fn get_version(self: &Device) -> String;
        fn get_name(self: &Device) -> String;
    }
}

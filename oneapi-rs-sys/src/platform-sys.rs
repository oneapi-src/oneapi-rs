//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::platform")]
pub mod ffi {
    // This is a workaround - cxx currently doesn't support passing
    // around vectors of pointers directly
    // https://github.com/dtolnay/cxx/issues/774#issuecomment-808674945
    struct PlatformPtr {
        ptr: UniquePtr<Platform>
    }

    // We're trying to import a DevicePtr defined in device-sys.rs.
    // This syntax will be refactored in a further cxx pull request,
    // see https://github.com/dtolnay/cxx/issues/297
    // and https://github.com/dtolnay/cxx/issues/353
    #[namespace = "sycl_shims::device"]
    extern "C++" {
        include!("oneapi-rs-sys/src/device-sys.rs.h");
        type DevicePtr = crate::device::ffi::DevicePtr;
    }

    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/platform.hpp");

        #[namespace = "sycl_shims"]
        type Device = crate::opaque::ffi::Device;
        #[namespace = "sycl_shims"]
        type Platform = crate::opaque::ffi::Platform;

        fn get_platforms() -> Vec<PlatformPtr>;

        fn get_devices(platform: &Platform) -> Vec<DevicePtr>;

        fn get_version(platform: &Platform) -> String;
        fn get_name(platform: &Platform) -> String;
        fn get_vendor(platform: &Platform) -> String;
    }
}

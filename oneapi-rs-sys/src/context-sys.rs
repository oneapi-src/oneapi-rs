//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::context")]
pub mod ffi {
    #[namespace = "sycl_shims"]
    extern "C++" {
        include!("oneapi-rs-sys/src/types-sys.rs.h");
        type DevicePtr = crate::types::ffi::DevicePtr;
    }

    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/context.hpp");

        #[namespace = "sycl_shims"]
        type Context = crate::types::ffi::Context;

        fn new_context(devices: Vec<DevicePtr>) -> UniquePtr<Context>;
    }
}

//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::usm")]
pub mod ffi {
    unsafe extern "C++" {
        #[namespace = "sycl_shims"]
        type Queue = crate::types::ffi::Queue;
    }

    extern "C++" {
        include!("oneapi-rs-sys/include/usm.hpp");
        unsafe fn aligned_alloc_device(alignment: usize, num_bytes: usize, queue: &Queue) -> Result<*mut u8>;
        unsafe fn aligned_alloc_host(alignment: usize, num_bytes: usize, queue: &Queue) -> Result<*mut u8>;
        unsafe fn aligned_alloc_shared(alignment: usize, num_bytes: usize, queue: &Queue) -> Result<*mut u8>;
        unsafe fn free(ptr: *mut u8, queue: &Queue);
    }
}

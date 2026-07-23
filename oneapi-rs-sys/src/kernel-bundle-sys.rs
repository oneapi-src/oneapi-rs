//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::kernel_bundle")]
pub mod ffi {
    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/kernel-bundle.hpp");

        #[namespace = "sycl_shims"]
        type Context = crate::types::ffi::Context;

        #[namespace = "sycl_shims"]
        type SourceKernelBundle = crate::types::ffi::SourceKernelBundle;

        #[namespace = "sycl_shims"]
        type ExecutableKernelBundle = crate::types::ffi::ExecutableKernelBundle;

        #[namespace = "sycl_shims"]
        type Kernel = crate::types::ffi::Kernel;

        fn create_kernel_bundle_from_source(ctxt: &Context, source: &str)
            -> UniquePtr<SourceKernelBundle>;
        fn build(source: &mut UniquePtr<SourceKernelBundle>) -> UniquePtr<ExecutableKernelBundle>;
        fn get_kernel(bundle: &mut UniquePtr<ExecutableKernelBundle>, name: &str) -> UniquePtr<Kernel>;
    }
}

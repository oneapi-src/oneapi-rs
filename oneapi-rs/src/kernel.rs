//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use bytemuck::Pod;
use oneapi_rs_sys::{kernel_bundle::ffi, types};

/// A kernel bundle which stores loaded SYCL source code.
pub struct SourceKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::SourceKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::SourceKernelBundle>> for SourceKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::SourceKernelBundle>) -> Self {
        Self(value)
    }
}

impl SourceKernelBundle {
    pub fn build(&mut self) -> ExecutableKernelBundle {
        ffi::build(&mut self.0).into()
    }
}

/// A kernel bundle which stores compiled SYCL kernels.
pub struct ExecutableKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::ExecutableKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::ExecutableKernelBundle>> for ExecutableKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::ExecutableKernelBundle>) -> Self {
        Self(value)
    }
}

impl ExecutableKernelBundle {
    pub fn get_kernel(&mut self, name: &str) -> Kernel {
        ffi::get_kernel(&mut self.0, name).into()
    }
}

/// An executable SYCL kernel.
pub struct Kernel(pub(crate) cxx::UniquePtr<types::ffi::Kernel>);

impl From<cxx::UniquePtr<types::ffi::Kernel>> for Kernel {
    fn from(value: cxx::UniquePtr<types::ffi::Kernel>) -> Self {
        Self(value)
    }
}

/// Types which can be passed as SYCL kernel arguments.
pub unsafe trait KernelArgument {
    unsafe fn as_raw_arg(&self) -> &[u8];
}

unsafe impl<T: Pod> KernelArgument for T {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Types which describe an argument list for a SYCL kernel.
pub unsafe trait KernelArgumentList<const ARGC: usize> {
    unsafe fn as_raw_arg_list(&self) -> [&[u8]; ARGC];
}

unsafe impl<T: KernelArgument> KernelArgumentList<1> for T {
    unsafe fn as_raw_arg_list(&self) -> [&[u8]; 1] {
        [unsafe { self.as_raw_arg() }]
    }
}

pub use oneapi_rs_derive::KernelArgumentList;

use oneapi_rs_derive::impl_arg_list_for_tuples;

impl_arg_list_for_tuples! {16}

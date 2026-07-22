//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use bytemuck::Pod;
use oneapi_rs_sys::{kernel_bundle::ffi, types};

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

pub struct ExecutableKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::ExecutableKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::ExecutableKernelBundle>> for ExecutableKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::ExecutableKernelBundle>) -> Self {
        Self(value)
    }
}

pub unsafe trait KernelArgument {
    unsafe fn as_raw_arg(&self) -> &[u8];
}

unsafe impl<T: Pod> KernelArgument for T {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

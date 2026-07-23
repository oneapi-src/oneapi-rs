//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use bytemuck::Pod;
use oneapi_rs_sys::{kernel_bundle::ffi, types};

use crate::{event::Event, queue::Queue};

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

impl ExecutableKernelBundle {
    pub fn get_kernel(&mut self, name: &str) -> Kernel {
        ffi::get_kernel(&mut self.0, name).into()
    }
}

pub struct Kernel(pub(crate) cxx::UniquePtr<types::ffi::Kernel>);

impl From<cxx::UniquePtr<types::ffi::Kernel>> for Kernel {
    fn from(value: cxx::UniquePtr<types::ffi::Kernel>) -> Self {
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

pub unsafe trait KernelArgumentList<const ARGC: usize> {
    unsafe fn as_raw_arg_list(&self) -> [&[u8]; ARGC];
}

pub type Range<const DIMENSIONS: usize = 1> = [u64; DIMENSIONS];

pub struct NdRange<const DIMENSIONS: usize = 1> {
    pub group_size: Range<DIMENSIONS>,
    pub local_size: Range<DIMENSIONS>,
}

impl<const DIMENSIONS: usize> NdRange<DIMENSIONS> {
    pub fn new(group_size: Range<DIMENSIONS>, local_size: Range<DIMENSIONS>) -> Self {
        Self {
            group_size,
            local_size,
        }
    }
}

pub trait ValidDimension {
    unsafe fn launch<const ARGC: usize>(
        &self,
        queue: &mut Queue,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Event;
}

impl ValidDimension for NdRange<1> {
    unsafe fn launch<const ARGC: usize>(
        &self,
        queue: &mut Queue,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Event {
        unsafe {
            oneapi_rs_sys::queue::ffi::launch_1d(
                &mut queue.0,
                types::ffi::Range1 {
                    data: self.group_size,
                },
                types::ffi::Range1 {
                    data: self.local_size,
                },
                &kernel.0,
                &args.as_raw_arg_list(),
            )
        }
        .into()
    }
}

impl ValidDimension for NdRange<2> {
    unsafe fn launch<const ARGC: usize>(
        &self,
        queue: &mut Queue,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Event {
        unsafe {
            oneapi_rs_sys::queue::ffi::launch_2d(
                &mut queue.0,
                types::ffi::Range2 {
                    data: self.group_size,
                },
                types::ffi::Range2 {
                    data: self.local_size,
                },
                &kernel.0,
                &args.as_raw_arg_list(),
            )
        }
        .into()
    }
}

impl ValidDimension for NdRange<3> {
    unsafe fn launch<const ARGC: usize>(
        &self,
        queue: &mut Queue,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Event {
        unsafe {
            oneapi_rs_sys::queue::ffi::launch_3d(
                &mut queue.0,
                types::ffi::Range3 {
                    data: self.group_size,
                },
                types::ffi::Range3 {
                    data: self.local_size,
                },
                &kernel.0,
                &args.as_raw_arg_list(),
            )
        }
        .into()
    }
}

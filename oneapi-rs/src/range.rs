//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs_sys::types;

use crate::{
    event::Event,
    kernel::{Kernel, KernelArgumentList},
    queue::Queue,
};

/// `Range` is a 1D, 2D or 3D vector that defines the iteration domain of either a single work-group
/// in a parallel dispatch, or the overall dimensions of the dispatch.
pub type Range<const DIMENSIONS: usize = 1> = [u64; DIMENSIONS];

/// The `NdRange` struct defines the iteration domain of both the work-groups and the overall
/// dispatch.
///
/// An `NdRange` comprises two [`Range`] parameters: the whole range over which the kernel is to be
/// executed and the range of each work group.
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

/// [`NdRange`] types which are limited to 1, 2 or 3 dimensions.
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

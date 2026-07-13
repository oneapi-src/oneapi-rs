//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs_sys::queue::ffi;

use crate::{buffer::Buffer, device::Device, usm::{HostAllocator, SharedAllocator, UsmAllocator}};

/// The `Queue` connects a host program to a single device. Programs submit tasks to a device via the
/// `Queue` and may monitor the `Queue` for completion. A program initiates the task by submitting
/// a kernel.
pub struct Queue(pub(crate) cxx::UniquePtr<ffi::Queue>);

impl Queue {
    /// Construct a `Queue` based on the device returned from the default selector.
    pub fn new() -> Self {
        Self(ffi::new_queue())
    }

    pub unsafe fn alloc_uninit_host<T>(&self, len: usize) -> Buffer<T, UsmAllocator<'_, HostAllocator>> {
        let allocator = UsmAllocator::from(self);
        unsafe { Buffer::new(allocator, len) }
    }

    pub unsafe fn alloc_uninit_shared<T>(&self, len: usize) -> Buffer<T, UsmAllocator<'_, SharedAllocator>> {
        let allocator = UsmAllocator::from(self);
        unsafe { Buffer::new(allocator, len) }
    }
}

impl From<&Device> for Queue {
    fn from(value: &Device) -> Self {
        Self(ffi::new_queue_from_device(&value.0))
    }
}

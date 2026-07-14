//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::queue::Queue;

use oneapi_rs_sys::usm::ffi;
use allocator_api2::alloc::{Allocator, AllocError};

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

type CxxResult<T> = cxx::core::result::Result<T, cxx::Exception>;

/// An instance of a USM allocator.
pub struct UsmAllocator<'a, T: UsmAllocatorKind> {
    queue: &'a Queue,
    _kind: PhantomData<T>
}

/// A marker trait for USM allocators.
pub unsafe trait UsmAlloc : Allocator {}

unsafe impl<'a, T: UsmAllocatorKind> UsmAlloc for UsmAllocator<'a, T> {}

pub trait UsmAllocatorKind {
    unsafe fn alloc(alignment: usize, num_bytes: usize, queue: &Queue) -> CxxResult<*mut u8>;
}

impl<'a, T: UsmAllocatorKind> From<&'a Queue> for UsmAllocator<'a, T> {
    fn from(queue: &'a Queue) -> Self {
        Self {
            queue,
            _kind: PhantomData
        }
    }
}

unsafe impl<T: UsmAllocatorKind> Allocator for UsmAllocator<'_, T> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = unsafe { T::alloc(layout.align(), layout.size(), &self.queue) }
            .map_err(|_e| AllocError)?;

        let ptr = NonNull::new(ptr).ok_or(AllocError)?;
        let slice = NonNull::slice_from_raw_parts(ptr, layout.size());

        Ok(slice)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        unsafe { ffi::free(ptr.as_ptr(), &self.queue.0); }
    }
}

/// An allocator for Device-side buffers
/// Safety: memory allocated by this allocator cannot be accessed on the host side
#[allow(dead_code)]
pub(crate) struct DeviceAllocator;

impl UsmAllocatorKind for DeviceAllocator {
    unsafe fn alloc(alignment: usize, num_bytes: usize, queue: &Queue) -> CxxResult<*mut u8> {
        unsafe { ffi::aligned_alloc_device(alignment, num_bytes, &queue.0) }
    }
}

/// An allocator for Host-side buffers
pub struct HostAllocator;

impl UsmAllocatorKind for HostAllocator {
    unsafe fn alloc(alignment: usize, num_bytes: usize, queue: &Queue) -> CxxResult<*mut u8> {
        unsafe { ffi::aligned_alloc_host(alignment, num_bytes, &queue.0) }
    }
}

/// An allocator for shared memory buffers
pub struct SharedAllocator;

impl UsmAllocatorKind for SharedAllocator {
    unsafe fn alloc(alignment: usize, num_bytes: usize, queue: &Queue) -> CxxResult<*mut u8> {
        unsafe { ffi::aligned_alloc_shared(alignment, num_bytes, &queue.0) }
    }
}

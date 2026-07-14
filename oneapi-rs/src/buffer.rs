//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{alloc::{Layout, handle_alloc_error}, ops::{Deref, DerefMut}, ptr::NonNull, slice};

use crate::usm::UsmAlloc;

/// The Buffer struct defines a shared array of one, two or three dimensions that can be used
/// by the SYCL kernel. Buffers are templated on the type of their data, and the number of
/// dimensions that the data is stored and accessed through.
///
/// A Buffer does not map to only one underlying backend object, and all SYCL backend memory objects
/// may be temporary for use on a specific device.
///
/// Buffers can be constructed by methods provided by the [`Queue`](`crate::queue::Queue`) class.
///
/// The Buffer struct template takes a template parameter [`UsmAlloc`](`crate::usm::UsmAlloc`) for
/// specifying an allocator which is used by the SYCL runtime when allocating temporary memory on
/// the host.
pub struct Buffer<T, A: UsmAlloc> {
    data: NonNull<T>,
    len: usize,
    layout: Layout,
    allocator: A,
}

impl<T, A: UsmAlloc> Buffer<T, A> {
    /// Creates a new buffer given an allocator.
    /// Safety: returns uninitialized memory.
    pub(crate) unsafe fn new(allocator: A, len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = match allocator.allocate(layout.clone()) {
            Ok(ptr) => ptr,
            _ => handle_alloc_error(layout)
        };

        Self {
            data: ptr.cast(),
            len,
            layout,
            allocator,
        }
    }
}

impl<T, A: UsmAlloc> Deref for Buffer<T, A> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(self.data.as_ptr(), self.len)
        }
    }
}

impl<T, A: UsmAlloc> DerefMut for Buffer<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
        }
    }
}

impl<T, A: UsmAlloc> Drop for Buffer<T, A> {
    fn drop(&mut self) {
        unsafe { self.allocator.deallocate(self.data.cast(), self.layout); }
    }
}

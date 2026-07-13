//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{alloc::{Layout, handle_alloc_error}, ops::{Deref, DerefMut}, ptr::NonNull, slice};

use crate::usm::UsmAlloc;

pub struct Buffer<T, A: UsmAlloc> {
    data: NonNull<T>,
    len: usize,
    layout: Layout,
    allocator: A,
}

impl<T, A: UsmAlloc> Buffer<T, A> {
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

//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{alloc::{Layout, handle_alloc_error}, ops::{Deref, DerefMut}, ptr::NonNull};

use crate::usm::UsmAlloc;

pub struct Buffer<T, const N: usize, A: UsmAlloc> {
    data: NonNull<[T; N]>,
    layout: Layout,
    allocator: A,
}

impl<T, const N: usize, A: UsmAlloc> Buffer<T, N, A> {
    pub(crate) fn new(allocator: A) -> Self {
        let layout = Layout::new::<[T; N]>();
        let ptr = match allocator.allocate(layout.clone()) {
            Ok(ptr) => ptr,
            _ => handle_alloc_error(layout)
        };

        Self {
            data: ptr.cast(),
            layout,
            allocator,
        }
    }
}

impl<T, const N: usize, A: UsmAlloc> Deref for Buffer<T, N, A> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe {
            self.data.as_ref()
        }
    }
}

impl<T, const N: usize, A: UsmAlloc> DerefMut for Buffer<T, N, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            self.data.as_mut()
        }
    }
}

impl<T, const N: usize, A: UsmAlloc> Drop for Buffer<T, N, A> {
    fn drop(&mut self) {
        unsafe { self.allocator.deallocate(self.data.cast(), self.layout); }
    }
}

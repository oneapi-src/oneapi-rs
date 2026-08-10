//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{
    alloc::{Layout, handle_alloc_error},
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::NonNull,
    slice,
    task::{Context, Poll},
};

use bytemuck::Pod;
use pin_project::pin_project;

use crate::{
    Result,
    event::{Event, EventFuture},
    kernel::KernelArgument,
    usm::{
        DeviceAllocator, HostAccessible, HostAllocator, SharedAllocator, UsmAlloc, UsmAllocator,
    },
};

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
            _ => handle_alloc_error(layout),
        };

        Self {
            data: ptr.cast(),
            len,
            layout,
            allocator,
        }
    }

    pub(crate) fn get_byte_ptr(&self) -> *mut u8 {
        self.data.as_ptr().cast()
    }

    pub(crate) fn get_byte_size(&self) -> usize {
        self.layout.size()
    }

    pub(crate) fn get_len(&self) -> usize {
        self.len
    }

    unsafe fn as_raw_arg_impl(&self) -> &[u8] {
        let data_ptr: *const NonNull<_> = &self.data;
        let cast_ptr = data_ptr as *const u8;
        unsafe { slice::from_raw_parts(cast_ptr, std::mem::size_of_val(&cast_ptr)) }
    }
}

impl<T, A: UsmAlloc + HostAccessible> Deref for Buffer<T, A> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl<T, A: UsmAlloc + HostAccessible> DerefMut for Buffer<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl<T, A: UsmAlloc> Drop for Buffer<T, A> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.deallocate(self.data.cast(), self.layout);
        }
    }
}

pub type HostBuffer<T> = Buffer<T, UsmAllocator<HostAllocator>>;
pub type SharedBuffer<T> = Buffer<T, UsmAllocator<SharedAllocator>>;
pub type DeviceBuffer<T> = Buffer<T, UsmAllocator<DeviceAllocator>>;

/// A [`Buffer`] whose initialization has been enqueued. You need to wait/await it.
pub struct EnqueuedBuffer<T, A: UsmAlloc> {
    buffer: Buffer<T, A>,
    event: Event,
}

impl<T, A: UsmAlloc> EnqueuedBuffer<T, A> {
    pub(crate) fn new(buffer: Buffer<T, A>, event: Event) -> Self {
        Self { buffer, event }
    }
}

impl<T, A: UsmAlloc> EnqueuedBuffer<T, A> {
    /// Performs a blocking wait for the [`Buffer`] initialization to complete. Returns an error if
    /// a synchronous SYCL exception occurs.
    pub fn wait(mut self) -> Result<Buffer<T, A>> {
        self.event.wait().map(|_| self.buffer)
    }
}

pub type EnqueuedHostBuffer<T> = EnqueuedBuffer<T, UsmAllocator<HostAllocator>>;
pub type EnqueuedSharedBuffer<T> = EnqueuedBuffer<T, UsmAllocator<SharedAllocator>>;
pub type EnqueuedDeviceBuffer<T> = EnqueuedBuffer<T, UsmAllocator<DeviceAllocator>>;

#[pin_project]
/// A [`Future`] which represents a pending [`Buffer`] allocation.
pub struct BufferFuture<T, A: UsmAlloc> {
    buffer: Option<Buffer<T, A>>,
    #[pin]
    event_future: EventFuture,
}

impl<T, A: UsmAlloc> Future for BufferFuture<T, A> {
    type Output = Result<Buffer<T, A>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        this.event_future
            .poll(cx)
            .map(|result| result.map(|_| this.buffer.take().unwrap()))
    }
}

impl<T, A: UsmAlloc> IntoFuture for EnqueuedBuffer<T, A> {
    type Output = Result<Buffer<T, A>>;
    type IntoFuture = BufferFuture<T, A>;

    fn into_future(self) -> Self::IntoFuture {
        Self::IntoFuture {
            buffer: Some(self.buffer),
            event_future: self.event.into_future(),
        }
    }
}

pub type HostBufferFuture<T> = BufferFuture<T, UsmAllocator<HostAllocator>>;
pub type SharedBufferFuture<T> = BufferFuture<T, UsmAllocator<SharedAllocator>>;
pub type DeviceBufferFuture<T> = BufferFuture<T, UsmAllocator<DeviceAllocator>>;

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for Buffer<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for &Buffer<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for &mut Buffer<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

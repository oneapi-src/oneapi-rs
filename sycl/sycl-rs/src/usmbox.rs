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

/// The `UsmBox` struct defines a shared array of one, two or three dimensions that can be used
/// by the SYCL kernel. `UsmBox`es are generic w.r.t. the type of their data, and the number of
/// dimensions that the data is stored and accessed through.
///
/// A `UsmBox` does not map to only one underlying backend object, and all SYCL backend memory objects
/// may be temporary for use on a specific device.
///
/// `UsmBox`es can be constructed by methods provided by the [`Queue`](`crate::queue::Queue`) class.
///
/// The `UsmBox` struct takes a generic parameter [`UsmAlloc`](`crate::usm::UsmAlloc`) for
/// specifying an allocator which is used by the SYCL runtime when allocating temporary memory on
/// the host.
pub struct UsmBox<T, A: UsmAlloc> {
    data: NonNull<T>,
    len: usize,
    layout: Layout,
    allocator: A,
}

impl<T, A: UsmAlloc> UsmBox<T, A> {
    /// Creates a new array given an allocator.
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

impl<T, A: UsmAlloc + HostAccessible> Deref for UsmBox<T, A> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl<T, A: UsmAlloc + HostAccessible> DerefMut for UsmBox<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl<T, A: UsmAlloc> Drop for UsmBox<T, A> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.deallocate(self.data.cast(), self.layout);
        }
    }
}

pub type HostUsmBox<T> = UsmBox<T, UsmAllocator<HostAllocator>>;
pub type SharedUsmBox<T> = UsmBox<T, UsmAllocator<SharedAllocator>>;
pub type DeviceUsmBox<T> = UsmBox<T, UsmAllocator<DeviceAllocator>>;

/// A [`UsmBox`] whose initialization has been enqueued. You need to wait/await it.
pub struct EnqueuedUsmBox<T, A: UsmAlloc> {
    array: UsmBox<T, A>,
    event: Event,
}

impl<T, A: UsmAlloc> EnqueuedUsmBox<T, A> {
    pub(crate) fn new(array: UsmBox<T, A>, event: Event) -> Self {
        Self { array, event }
    }
}

impl<T, A: UsmAlloc> EnqueuedUsmBox<T, A> {
    /// Performs a blocking wait for the [`UsmBox`] initialization to complete. Returns an error if
    /// a synchronous SYCL exception occurs.
    ///
    /// Dropping an enqueued array does not wait for its completion.
    pub fn wait(mut self) -> Result<UsmBox<T, A>> {
        self.event.wait().map(|_| self.array)
    }
}

pub type EnqueuedHostUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<HostAllocator>>;
pub type EnqueuedSharedUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<SharedAllocator>>;
pub type EnqueuedDeviceUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<DeviceAllocator>>;

#[pin_project]
/// A [`Future`] which represents a pending [`UsmBox`] allocation.
pub struct UsmBoxFuture<T, A: UsmAlloc> {
    array: Option<UsmBox<T, A>>,
    #[pin]
    event_future: EventFuture,
}

impl<T, A: UsmAlloc> Future for UsmBoxFuture<T, A> {
    type Output = Result<UsmBox<T, A>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        this.event_future
            .poll(cx)
            .map(|result| result.map(|_| this.array.take().unwrap()))
    }
}

impl<T, A: UsmAlloc> IntoFuture for EnqueuedUsmBox<T, A> {
    type Output = Result<UsmBox<T, A>>;
    type IntoFuture = UsmBoxFuture<T, A>;

    fn into_future(self) -> Self::IntoFuture {
        Self::IntoFuture {
            array: Some(self.array),
            event_future: self.event.into_future(),
        }
    }
}

pub type HostUsmBoxFuture<T> = UsmBoxFuture<T, UsmAllocator<HostAllocator>>;
pub type SharedUsmBoxFuture<T> = UsmBoxFuture<T, UsmAllocator<SharedAllocator>>;
pub type DeviceUsmBoxFuture<T> = UsmBoxFuture<T, UsmAllocator<DeviceAllocator>>;

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for UsmBox<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for &UsmBox<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

unsafe impl<T: Pod, A: UsmAlloc> KernelArgument for &mut UsmBox<T, A> {
    unsafe fn as_raw_arg(&self) -> &[u8] {
        unsafe { self.as_raw_arg_impl() }
    }
}

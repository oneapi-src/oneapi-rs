//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::cmp::min;

use bytemuck::Pod;
use oneapi_rs_sys::{queue::ffi, types::ffi::EventPtr};

use crate::{
    buffer::{
        Buffer, DeviceBuffer, EnqueuedBuffer, EnqueuedDeviceBuffer, EnqueuedHostBuffer,
        EnqueuedSharedBuffer, HostBuffer, SharedBuffer,
    },
    context::Context,
    device::Device,
    event::Event,
    kernel::{Kernel, KernelArgumentList},
    range::{NdRange, ValidDimension},
    usm::{UsmAlloc, UsmAllocator},
};

/// The `Queue` connects a host program to a single device. Programs submit tasks to a device via the
/// `Queue` and may monitor the `Queue` for completion. A program initiates the task by submitting
/// a kernel.
pub struct Queue(pub(crate) cxx::UniquePtr<ffi::Queue>);

impl Queue {
    /// Construct a `Queue` based on the device returned from the default selector.
    pub fn new() -> Self {
        Self(ffi::new_queue())
    }

    /// Construct an immediate `Queue` based on the device returned from the default selector.
    pub fn new_immediate() -> Self {
        Self(ffi::new_queue_immediate())
    }

    /// Returns the SYCL queue’s context.
    pub fn get_context(&self) -> Context {
        ffi::get_context(&self.0).into()
    }

    /// Allocates zeroed memory and creates a host-side [`Buffer`] that can store an array of T.
    pub fn alloc_host<T: Pod>(&mut self, len: usize) -> EnqueuedHostBuffer<T> {
        unsafe {
            let mut buffer = self.alloc_uninit_host(len);
            let event = self.memset(&mut buffer, 0);
            EnqueuedBuffer::new(buffer, event)
        }
    }

    /// Allocates zeroed memory and creates a shared [`Buffer`] that can store an array of T.
    pub fn alloc_shared<T: Pod>(&mut self, len: usize) -> EnqueuedSharedBuffer<T> {
        unsafe {
            let mut buffer = self.alloc_uninit_shared(len);
            let event = self.memset(&mut buffer, 0);
            EnqueuedBuffer::new(buffer, event)
        }
    }

    /// Allocates zeroed memory and creates a device [`Buffer`] that can store an array of T.
    pub fn alloc_device<T: Pod>(&mut self, len: usize) -> EnqueuedDeviceBuffer<T> {
        unsafe {
            let mut buffer = self.alloc_uninit_device(len);
            let event = self.memset(&mut buffer, 0);
            EnqueuedBuffer::new(buffer, event)
        }
    }

    /// Allocates memory and creates a host-side [`Buffer`] that can store an array of T.
    /// Safety: the buffer contents are uninitialized.
    pub unsafe fn alloc_uninit_host<T>(&self, len: usize) -> HostBuffer<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { Buffer::new(allocator, len) }
    }

    /// Allocates memory and creates a shared [`Buffer`] that can store an array of T.
    /// Safety: the buffer contents are uninitialized.
    pub unsafe fn alloc_uninit_shared<T>(&self, len: usize) -> SharedBuffer<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { Buffer::new(allocator, len) }
    }

    /// Allocates memory and creates a device-side [`Buffer`] that can store an array of T.
    /// Safety: the buffer contents are uninitialized.
    pub unsafe fn alloc_uninit_device<T>(&self, len: usize) -> DeviceBuffer<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { Buffer::new(allocator, len) }
    }

    /// Sets memory allocated with USM allocations.
    /// Safety: the caller must make sure the underlying memory isn't being aliased somewhere else.
    pub unsafe fn memset<T, A: UsmAlloc>(
        &mut self,
        buffer: &mut Buffer<T, A>,
        value: i32,
    ) -> Event {
        unsafe { self.memset_with_deps(buffer, value, &[]) }
    }

    /// Sets memory allocated with USM allocations after all specified events finish.
    /// Safety: the caller must make sure the underlying memory isn't being aliased somewhere else.
    pub unsafe fn memset_with_deps<T, A: UsmAlloc>(
        &mut self,
        buffer: &mut Buffer<T, A>,
        value: i32,
        dep_events: &[&Event],
    ) -> Event {
        let ptr = buffer.get_byte_ptr();
        let num_bytes = buffer.get_byte_size();
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();
        unsafe { ffi::memset(&mut self.0, ptr, value, num_bytes, dep_events) }.into()
    }

    /// Submits a barrier to the queue.
    pub fn barrier(&mut self) -> Event {
        self.barrier_with_deps(&[])
    }

    /// Submits a barrier to the queue after all specified events finish.
    pub fn barrier_with_deps(&mut self, dep_events: &[&Event]) -> Event {
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();
        ffi::barrier(&mut self.0, dep_events).into()
    }

    /// Performs a blocking wait for the completion of all enqueued tasks in the queue.
    pub fn wait(&mut self) {
        ffi::wait(&mut self.0);
    }

    /// Enqueues a kernel object to the queue as an ND-range kernel, using the number of work-items
    /// specified by the [`NdRange`] nd_range.
    pub unsafe fn launch<const ARGC: usize, const DIMENSIONS: usize>(
        &mut self,
        nd_range: NdRange<DIMENSIONS>,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Event
    where
        NdRange<DIMENSIONS>: ValidDimension,
    {
        unsafe { nd_range.launch(self, kernel, args) }
    }

    /// Copies the contents of the source buffer to the destination buffer.
    ///
    /// If the buffer sizes don't match the destination buffer is filled with as many elements from
    /// source buffer as possible.
    pub fn copy<T, A1, A2>(&mut self, src: &Buffer<T, A1>, dst: &mut Buffer<T, A2>) -> Event
    where
        T: Pod,
        A1: UsmAlloc,
        A2: UsmAlloc,
    {
        self.copy_with_deps(src, dst, &[])
    }

    /// Copies the contents of the source buffer to the destination buffer after all specified
    /// events finish.
    ///
    /// If the buffer sizes don't match the destination buffer is filled with as many elements from
    /// source buffer as possible.
    pub fn copy_with_deps<T, A1, A2>(
        &mut self,
        src: &Buffer<T, A1>,
        dst: &mut Buffer<T, A2>,
        dep_events: &[&Event],
    ) -> Event
    where
        T: Pod,
        A1: UsmAlloc,
        A2: UsmAlloc,
    {
        // TODO: Resolve the C++ lifetime elision issue
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();

        let amount = min(src.get_len(), dst.get_len());
        let num_bytes = amount * size_of::<T>();
        unsafe {
            ffi::memcpy(
                &mut self.0,
                dst.get_byte_ptr(),
                src.get_byte_ptr(),
                num_bytes,
                dep_events,
            )
        }
        .into()
    }
}

impl From<&Device> for Queue {
    fn from(value: &Device) -> Self {
        Self(ffi::new_queue_from_device(&value.0))
    }
}

impl From<cxx::UniquePtr<ffi::Queue>> for Queue {
    fn from(value: cxx::UniquePtr<ffi::Queue>) -> Self {
        Self(value)
    }
}

impl Clone for Queue {
    fn clone(&self) -> Self {
        ffi::clone(&self.0).into()
    }
}

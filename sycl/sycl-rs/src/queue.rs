//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::Result;
use bytemuck::Pod;
use sycl_rs_sys::{queue::ffi, types::ffi::EventPtr};

use crate::{
    context::Context,
    device::Device,
    event::Event,
    kernel::{Kernel, KernelArgumentList},
    range::{NdRange, ValidDimension},
    usm::{UsmAlloc, UsmAllocator},
    usmbox::{
        DeviceUsmBox, EnqueuedDeviceUsmBox, EnqueuedHostUsmBox, EnqueuedSharedUsmBox,
        EnqueuedUsmBox, HostUsmBox, SharedUsmBox, UsmBox,
    },
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

    /// Allocates zeroed memory and creates a host-side [`UsmBox`] that can store an array of T.
    pub fn alloc_host<T: Pod>(&mut self, len: usize) -> Result<EnqueuedHostUsmBox<T>> {
        unsafe {
            let mut array = self.alloc_uninit_host(len);
            self.memset(&mut array, 0)
                .map(|event| EnqueuedUsmBox::new(array, event))
        }
    }

    /// Allocates zeroed memory and creates a shared [`UsmBox`] that can store an array of T.
    pub fn alloc_shared<T: Pod>(&mut self, len: usize) -> Result<EnqueuedSharedUsmBox<T>> {
        unsafe {
            let mut array = self.alloc_uninit_shared(len);
            self.memset(&mut array, 0)
                .map(|event| EnqueuedUsmBox::new(array, event))
        }
    }

    /// Allocates zeroed memory and creates a device [`UsmBox`] that can store an array of T.
    pub fn alloc_device<T: Pod>(&mut self, len: usize) -> Result<EnqueuedDeviceUsmBox<T>> {
        unsafe {
            let mut array = self.alloc_uninit_device(len);
            self.memset(&mut array, 0)
                .map(|event| EnqueuedUsmBox::new(array, event))
        }
    }

    /// Allocates memory and creates a host-side [`UsmBox`] that can store an array of T.
    /// Safety: the array contents are uninitialized.
    pub unsafe fn alloc_uninit_host<T>(&self, len: usize) -> HostUsmBox<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { UsmBox::new(allocator, len) }
    }

    /// Allocates memory and creates a shared [`UsmBox`] that can store an array of T.
    /// Safety: the array contents are uninitialized.
    pub unsafe fn alloc_uninit_shared<T>(&self, len: usize) -> SharedUsmBox<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { UsmBox::new(allocator, len) }
    }

    /// Allocates memory and creates a device-side [`UsmBox`] that can store an array of T.
    /// Safety: the array contents are uninitialized.
    pub unsafe fn alloc_uninit_device<T>(&self, len: usize) -> DeviceUsmBox<T> {
        let allocator = UsmAllocator::from(self);
        unsafe { UsmBox::new(allocator, len) }
    }

    /// Sets memory allocated with USM allocations.
    /// Safety: the caller must make sure the underlying memory isn't being aliased somewhere else.
    pub unsafe fn memset<T, A: UsmAlloc>(
        &mut self,
        array: &mut UsmBox<T, A>,
        value: i32,
    ) -> Result<Event> {
        unsafe { self.memset_with_deps(array, value, &[]) }
    }

    /// Sets memory allocated with USM allocations after all specified events finish.
    /// Safety: the caller must make sure the underlying memory isn't being aliased somewhere else.
    pub unsafe fn memset_with_deps<T, A: UsmAlloc>(
        &mut self,
        array: &mut UsmBox<T, A>,
        value: i32,
        dep_events: &[&Event],
    ) -> Result<Event> {
        let ptr = array.get_byte_ptr();
        let num_bytes = array.get_byte_size();
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();
        unsafe { ffi::memset(&mut self.0, ptr, value, num_bytes, dep_events) }.map(Into::into)
    }

    /// Submits a barrier to the queue.
    pub fn barrier(&mut self) -> Result<Event> {
        self.barrier_with_deps(&[]).map(Into::into)
    }

    /// Submits a barrier to the queue after all specified events finish.
    pub fn barrier_with_deps(&mut self, dep_events: &[&Event]) -> Result<Event> {
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();
        ffi::barrier(&mut self.0, dep_events).map(Into::into)
    }

    /// Performs a blocking wait for the completion of all enqueued tasks in the queue. Returns an
    /// error if a synchronous SYCL exception occurs.
    ///
    /// Dropping the queue does not wait for its completion.
    pub fn wait(&mut self) -> Result<()> {
        ffi::wait(&mut self.0)
    }

    /// Enqueues a kernel object to the queue as an ND-range kernel, using the number of work-items
    /// specified by the [`NdRange`] nd_range.
    ///
    /// Safety: The caller must make sure each argument matches the launched SYCL kernel's
    /// signature, including their respective size, layout and alignment.
    pub unsafe fn launch<const ARGC: usize, const DIMENSIONS: usize>(
        &mut self,
        nd_range: NdRange<DIMENSIONS>,
        kernel: &Kernel,
        args: impl KernelArgumentList<ARGC>,
    ) -> Result<Event>
    where
        NdRange<DIMENSIONS>: ValidDimension,
    {
        unsafe { nd_range.launch(self, kernel, args) }
    }

    /// Copies the contents of the source array to the destination array.
    ///
    /// Panics if the source and destination array lengths differ.
    pub fn copy<T, A1, A2>(&mut self, src: &UsmBox<T, A1>, dst: &mut UsmBox<T, A2>) -> Result<Event>
    where
        T: Pod,
        A1: UsmAlloc,
        A2: UsmAlloc,
    {
        self.copy_with_deps(src, dst, &[])
    }

    /// Copies the contents of the source array to the destination array after all specified
    /// events finish.
    ///
    /// Panics if the source and destination array lengths differ.
    pub fn copy_with_deps<T, A1, A2>(
        &mut self,
        src: &UsmBox<T, A1>,
        dst: &mut UsmBox<T, A2>,
        dep_events: &[&Event],
    ) -> Result<Event>
    where
        T: Pod,
        A1: UsmAlloc,
        A2: UsmAlloc,
    {
        assert_eq!(
            src.get_len(),
            dst.get_len(),
            "source and destination array lengths differ"
        );

        // TODO: Resolve the C++ lifetime elision issue
        let dep_events = dep_events
            .iter()
            .map(|e| EventPtr {
                ptr: (*e).clone().0,
            })
            .collect::<Vec<_>>();

        let num_bytes = src.get_len() * size_of::<T>();
        unsafe {
            ffi::memcpy(
                &mut self.0,
                dst.get_byte_ptr(),
                src.get_byte_ptr(),
                num_bytes,
                dep_events,
            )
        }
        .map(Into::into)
    }
}

impl From<&Device> for Queue {
    fn from(value: &Device) -> Self {
        Self(ffi::new_queue_from_device(&value.0))
    }
}

impl From<(&Context, &Device)> for Queue {
    fn from(value: (&Context, &Device)) -> Self {
        Self(ffi::new_queue_from_context_and_device(
            &value.0.0, &value.1.0,
        ))
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

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
        let ptr = match allocator.allocate(layout) {
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
///
/// Dropping an enqueued array blocks until initialization completes before freeing its memory.
/// If waiting fails, the allocation is leaked rather than freed while it may still be in use.
/// A failed wait during drop panics.
pub struct EnqueuedUsmBox<T, A: UsmAlloc> {
    array: Option<UsmBox<T, A>>,
    event: Event,
}

impl<T, A: UsmAlloc> EnqueuedUsmBox<T, A> {
    pub(crate) fn new(array: UsmBox<T, A>, event: Event) -> Self {
        Self {
            array: Some(array),
            event,
        }
    }

    /// Performs a blocking wait for the [`UsmBox`] initialization to complete. Returns an error if
    /// a synchronous SYCL exception occurs.
    ///
    /// If waiting fails, the allocation is leaked because completion cannot be guaranteed.
    pub fn wait(mut self) -> Result<UsmBox<T, A>> {
        let result = self.event.wait();
        self.finish(result)
    }

    fn finish(&mut self, result: Result<()>) -> Result<UsmBox<T, A>> {
        let array = self.array.take().expect("allocation already consumed");
        match result {
            Ok(()) => Ok(array),
            Err(error) => {
                // A failed wait does not guarantee that the device has stopped using the memory.
                std::mem::forget(array);
                Err(error)
            }
        }
    }
}

impl<T, A: UsmAlloc> Drop for EnqueuedUsmBox<T, A> {
    fn drop(&mut self) {
        if self.array.is_some() {
            let result = self.event.wait();
            drop(
                self.finish(result)
                    .expect("failed to wait for USM initialization during drop; allocation leaked"),
            );
        }
    }
}

pub type EnqueuedHostUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<HostAllocator>>;
pub type EnqueuedSharedUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<SharedAllocator>>;
pub type EnqueuedDeviceUsmBox<T> = EnqueuedUsmBox<T, UsmAllocator<DeviceAllocator>>;

#[pin_project]
/// A [`Future`] which represents a pending [`UsmBox`] allocation.
///
/// Dropping this future, even before its first poll, blocks until initialization completes.
/// If waiting fails during drop, the allocation is leaked and the drop panics.
pub struct UsmBoxFuture<T, A: UsmAlloc> {
    enqueued: EnqueuedUsmBox<T, A>,
    #[pin]
    event_future: EventFuture,
}

impl<T, A: UsmAlloc> Future for UsmBoxFuture<T, A> {
    type Output = Result<UsmBox<T, A>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        this.event_future
            .poll(cx)
            .map(|result| this.enqueued.finish(result))
    }
}

impl<T, A: UsmAlloc> IntoFuture for EnqueuedUsmBox<T, A> {
    type Output = Result<UsmBox<T, A>>;
    type IntoFuture = UsmBoxFuture<T, A>;

    fn into_future(self) -> Self::IntoFuture {
        Self::IntoFuture {
            event_future: self.event.clone().into_future(),
            enqueued: self,
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

#[cfg(test)]
mod tests {
    use super::*;
    use allocator_api2::alloc::{AllocError, Allocator};
    use std::{
        sync::{
            Arc, Condvar, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        task::{Wake, Waker},
        thread,
        time::Duration,
    };
    use sycl_rs_sys::{event::ffi, types::SharedWaker};

    #[derive(Default)]
    struct Gate {
        state: Mutex<(bool, bool)>,
        changed: Condvar,
    }

    impl Wake for Gate {
        fn wake(self: Arc<Self>) {
            let mut state = self.state.lock().unwrap();
            state.0 = true;
            self.changed.notify_all();
            while !state.1 {
                state = self.changed.wait(state).unwrap();
            }
        }
    }

    struct Release(Arc<Gate>);

    impl Drop for Release {
        fn drop(&mut self) {
            self.0.state.lock().unwrap().1 = true;
            self.0.changed.notify_all();
        }
    }

    struct ObservedAllocator<'a> {
        allocation: &'a HostUsmBox<u8>,
        event: &'a Event,
        deallocated: &'a AtomicBool,
        premature: &'a AtomicBool,
    }

    unsafe impl Allocator for ObservedAllocator<'_> {
        fn allocate(&self, layout: Layout) -> std::result::Result<NonNull<[u8]>, AllocError> {
            assert_eq!(layout.size(), self.allocation.get_byte_size());
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new(self.allocation.get_byte_ptr()).unwrap(),
                layout.size(),
            ))
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
            let complete = matches!(
                ffi::get_command_execution_status(&self.event.0),
                sycl_rs_sys::types::ffi::EventCommandStatus::Complete
            );
            self.premature.store(!complete, Ordering::SeqCst);
            self.deallocated.store(true, Ordering::SeqCst);
        }
    }

    unsafe impl UsmAlloc for ObservedAllocator<'_> {}

    #[derive(Clone, Copy)]
    enum Completion {
        DropEnqueued,
        DropUnpolled,
        DropPolled,
        Wait,
        Await,
    }

    fn check_initialization_ownership(completion: Completion) -> Result<()> {
        // Isolate fixtures from concurrent device discovery in the uninstrumented CPU runtime.
        static FIXTURE: Mutex<()> = Mutex::new(());
        let fixture = FIXTURE.lock().unwrap();
        let mut queue = crate::queue::Queue::new();
        let dependency = queue.barrier()?;
        let gate = Arc::new(Gate::default());
        let release = Release(gate.clone());
        let shared = Arc::new(SharedWaker::new());
        shared.waker.register(&Waker::from(gate.clone()));
        unsafe {
            ffi::register_callback(&mut queue.0, &dependency.0, Arc::into_raw(shared))?;
        }
        let (state, timeout) = gate
            .changed
            .wait_timeout_while(gate.state.lock().unwrap(), Duration::from_secs(10), |s| {
                !s.0
            })
            .unwrap();
        assert!(!timeout.timed_out(), "initialization gate did not start");
        drop(state);

        let mut allocation = unsafe { queue.alloc_uninit_host::<u8>(16) };
        let event = unsafe { queue.memset(&mut allocation, 0) }?;
        assert!(!matches!(
            ffi::get_command_execution_status(&event.0),
            sycl_rs_sys::types::ffi::EventCommandStatus::Complete
        ));
        let deallocated = AtomicBool::new(false);
        let premature = AtomicBool::new(false);
        // Keep real backing memory alive even if the ownership regression reappears.
        let array = unsafe {
            UsmBox::<u8, _>::new(
                ObservedAllocator {
                    allocation: &allocation,
                    event: &event,
                    deallocated: &deallocated,
                    premature: &premature,
                },
                16,
            )
        };
        let enqueued = EnqueuedUsmBox::new(array, event.clone());

        thread::scope(|scope| -> Result<()> {
            let (start, started) = mpsc::channel();
            scope.spawn(move || {
                let _release = release;
                started
                    .recv_timeout(Duration::from_secs(10))
                    .expect("completion path did not start");
                thread::sleep(Duration::from_millis(100));
            });
            match completion {
                Completion::DropEnqueued => {
                    start.send(()).unwrap();
                    drop(enqueued);
                }
                Completion::DropUnpolled => {
                    let future = enqueued.into_future();
                    start.send(()).unwrap();
                    drop(future);
                }
                Completion::DropPolled => {
                    let mut future = Box::pin(enqueued.into_future());
                    let mut context = Context::from_waker(Waker::noop());
                    assert!(future.as_mut().poll(&mut context).is_pending());
                    start.send(()).unwrap();
                    drop(future);
                }
                Completion::Wait => {
                    start.send(()).unwrap();
                    let array = enqueued.wait()?;
                    assert!(!deallocated.load(Ordering::SeqCst));
                    drop(array);
                }
                Completion::Await => {
                    start.send(()).unwrap();
                    let array = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap()
                        .block_on(enqueued.into_future())?;
                    assert!(!deallocated.load(Ordering::SeqCst));
                    drop(array);
                }
            }
            Ok(())
        })?;
        queue.wait()?;
        drop(fixture);
        assert!(deallocated.load(Ordering::SeqCst));
        assert!(!premature.load(Ordering::SeqCst));
        assert!(allocation.iter().all(|&value| value == 0));
        Ok(())
    }

    #[test]
    fn dropping_enqueued_allocation_waits_for_initialization() -> Result<()> {
        check_initialization_ownership(Completion::DropEnqueued)
    }

    #[test]
    fn dropping_unpolled_allocation_future_waits_for_initialization() -> Result<()> {
        check_initialization_ownership(Completion::DropUnpolled)
    }

    #[test]
    fn dropping_polled_allocation_future_waits_for_initialization() -> Result<()> {
        check_initialization_ownership(Completion::DropPolled)
    }

    #[test]
    fn waiting_returns_initialized_allocation() -> Result<()> {
        check_initialization_ownership(Completion::Wait)
    }

    #[test]
    fn awaiting_returns_initialized_allocation() -> Result<()> {
        check_initialization_ownership(Completion::Await)
    }
}

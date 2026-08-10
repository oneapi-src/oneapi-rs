//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{
    pin::Pin,
    sync::{Arc, atomic::Ordering::Relaxed},
    task::{Context, Poll},
};

use sycl_rs_sys::{event::ffi, types::SharedWaker};

use pin_project::pin_project;

use crate::{Result, info::InfoTarget, private::Sealed, queue::Queue};

pub struct Event(pub(crate) cxx::UniquePtr<ffi::Event>);

impl Event {
    /// Performs a blocking wait for the event to complete. Returns an error if a synchronous SYCL
    /// exception occurs.
    ///
    /// Dropping the event does not wait for its completion.
    pub fn wait(&mut self) -> Result<()> {
        ffi::wait(&mut self.0)
    }
}

impl Sealed for Event {}
impl InfoTarget for Event {}

impl From<cxx::UniquePtr<ffi::Event>> for Event {
    fn from(value: cxx::UniquePtr<ffi::Event>) -> Self {
        Self(value)
    }
}

impl Clone for Event {
    fn clone(&self) -> Self {
        ffi::clone(&self.0).into()
    }
}

#[pin_project]
pub struct EventFuture {
    event: Event,
    shared: Arc<SharedWaker>,
    set_callback: bool,
    queue: Option<Queue>,
}

impl Future for EventFuture {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        // Set the callback on first Future poll (Futures can't be active until polled)
        if *this.set_callback == false {
            *this.set_callback = true;
            let mut queue = Queue::new_immediate();
            this.shared.waker.register(cx.waker());

            // Safety: registered callback will decrement the Arc strong reference count, which is
            // large enough because it was increased by the previous clone().
            let ptr = Arc::into_raw(this.shared.clone());
            let result = unsafe { ffi::register_callback(&mut queue.0, &this.event.0, ptr) };
            match result {
                Ok(_) => {
                    this.queue.replace(queue);
                }
                Err(_) => {
                    return Poll::Ready(result);
                }
            }
        } else {
            // Quick check before registering to avoid wasting time
            if this.shared.done.load(Relaxed) {
                // The event finished - waiting for it returns immediately
                return Poll::Ready(this.event.wait());
            }

            this.shared.waker.register(cx.waker());
        }

        // Check the event again to avoid a race condition
        // https://docs.rs/futures/latest/futures/task/struct.AtomicWaker.html#examples
        if this.shared.done.load(Relaxed) {
            // The event finished - waiting for it returns immediately
            Poll::Ready(this.event.wait())
        } else {
            Poll::Pending
        }
    }
}

impl IntoFuture for Event {
    type Output = Result<()>;
    type IntoFuture = EventFuture;

    fn into_future(self) -> Self::IntoFuture {
        EventFuture {
            event: self,
            shared: Arc::new(SharedWaker::new()),
            set_callback: false,
            queue: None,
        }
    }
}

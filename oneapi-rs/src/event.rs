//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{pin::Pin, task::{Context, Poll}, sync::atomic::Ordering::Relaxed};

use oneapi_rs_sys::{event::ffi, types::SharedWaker};

use pin_project::pin_project;

use crate::{info::event::EventInfo, queue::Queue};

pub struct Event(pub(crate) cxx::UniquePtr<ffi::Event>);

impl Event {
    pub fn wait(&mut self) {
        ffi::wait(&mut self.0);
    }

    pub fn get_info<T: EventInfo>(&self) -> T::Item {
        T::get_item(self)
    }
}

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
    shared: SharedWaker,
    set_callback: bool,
    queue: Queue,
}

impl Future for EventFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        // Set the callback on first Future poll (Futures can't be active until polled)
        if *this.set_callback == false {
            *this.set_callback = true;
            this.shared.waker.register(cx.waker());
            unsafe { ffi::register_callback(&mut this.queue.0, &this.event.0, this.shared) };

            // Check the event again to avoid a race condition
            // https://docs.rs/futures/latest/futures/task/struct.AtomicWaker.html#examples
            if this.shared.done.load(Relaxed) {
                Poll::Ready(())
            }
            else {
                Poll::Pending
            }
        }
        else {
            // Quick check before registering to avoid wasting time
            if this.shared.done.load(Relaxed) {
                return Poll::Ready(());
            }

            // Register the waker if result isn't ready. This is a slow atomic operation
            this.shared.waker.register(cx.waker());

            // Check the event again to avoid a race condition
            // https://docs.rs/futures/latest/futures/task/struct.AtomicWaker.html#examples
            if this.shared.done.load(Relaxed) {
                Poll::Ready(())
            }
            else {
                Poll::Pending
            }

        }
    }
}

impl IntoFuture for Event {
    type Output = ();
    type IntoFuture = EventFuture;

    fn into_future(self) -> Self::IntoFuture {
        EventFuture {
            event: self,
            shared: SharedWaker::new(),
            set_callback: false,
            queue: Queue::new_immediate(),
        }
    }
}

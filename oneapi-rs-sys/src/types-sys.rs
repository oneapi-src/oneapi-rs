//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::sync::atomic::{Ordering::Relaxed, AtomicBool};

use futures::task::AtomicWaker;

pub struct SharedWaker {
    pub waker: AtomicWaker,
    pub done: AtomicBool
}

impl SharedWaker {
    pub fn new() -> Self {
        Self {
            waker: AtomicWaker::new(),
            done: AtomicBool::new(false)
        }
    }

    pub fn wake(&self) {
        self.done.store(true, Relaxed);
        self.waker.wake();
    }
}

#[cxx::bridge(namespace = "sycl_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/types.hpp");
        type Device;
        type Platform;
        type Queue;
        type Event;
    }

    // This is a workaround - cxx currently doesn't support passing
    // around vectors of pointers directly
    // https://github.com/dtolnay/cxx/issues/774#issuecomment-808674945
    // We must use pointer wrapper structs instead.
    struct DevicePtr {
        ptr: UniquePtr<Device>
    }

    struct PlatformPtr {
        ptr: UniquePtr<Platform>
    }
    
    struct EventPtr {
        ptr: UniquePtr<Event>
    }

    #[derive(Debug)]
    enum DeviceType {
        Cpu,
        Gpu,
        Accelerator,
        Custom,
        Automatic,
        All,
        Unimplemented
    }

    #[derive(Debug)]
    enum EventCommandStatus {
        Submitted,
        Running,
        Complete,
        Unknown
    }

    impl UniquePtr<Device> {}
    impl UniquePtr<Platform> {}
    impl UniquePtr<Queue> {}
    impl UniquePtr<Event> {}

    impl Vec<DevicePtr> {}
    impl Vec<PlatformPtr> {}
    impl Vec<EventPtr> {}
}

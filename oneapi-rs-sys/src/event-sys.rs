//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::types::SharedWaker;

#[cxx::bridge(namespace = "sycl_shims::event")]
pub mod ffi {
    #[namespace = "sycl_shims"]
    extern "C++" {
        include!("oneapi-rs-sys/src/types-sys.rs.h");
        type EventCommandStatus = crate::types::ffi::EventCommandStatus;
    }

    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/event.hpp");

        #[namespace = "sycl_shims"]
        type Event = crate::types::ffi::Event;

        #[namespace = "sycl_shims"]
        type Queue = crate::types::ffi::Queue;

        fn wait(event: &mut UniquePtr<Event>);
        unsafe fn register_callback(queue: &mut UniquePtr<Queue>, event: &Event, waker: *const SharedWaker);
        fn get_command_execution_status(event: &Event) -> EventCommandStatus;
        fn clone(event: &Event) -> UniquePtr<Event>;
    }

    extern "Rust" {
        type SharedWaker;
        fn wake(&self);
    }
}

//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[cxx::bridge(namespace = "sycl_shims::queue")]
pub mod ffi {
    #[namespace = "sycl_shims"]
    extern "C++" {
        include!("oneapi-rs-sys/src/types-sys.rs.h");
        type EventPtr = crate::types::ffi::EventPtr;
        type Range1 = crate::types::ffi::Range1;
        type Range2 = crate::types::ffi::Range2;
        type Range3 = crate::types::ffi::Range3;
    }

    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/queue.hpp");

        #[namespace = "sycl_shims"]
        type Queue = crate::types::ffi::Queue;
        #[namespace = "sycl_shims"]
        type Device = crate::types::ffi::Device;
        #[namespace = "sycl_shims"]
        type Context = crate::types::ffi::Context;
        #[namespace = "sycl_shims"]
        type Event = crate::types::ffi::Event;
        #[namespace = "sycl_shims"]
        type Kernel = crate::types::ffi::Kernel;

        fn new_queue() -> UniquePtr<Queue>;
        fn new_queue_immediate() -> UniquePtr<Queue>;
        fn new_queue_from_device(device: &Device) -> UniquePtr<Queue>;
        fn get_context(queue: &Queue) -> UniquePtr<Context>;
        fn clone(queue: &Queue) -> UniquePtr<Queue>;

        unsafe fn memset(
            queue: &mut UniquePtr<Queue>,
            ptr: *mut u8,
            value: i32,
            num_bytes: usize,
            dep_events: Vec<EventPtr>,
        ) -> UniquePtr<Event>;

        fn barrier(queue: &mut UniquePtr<Queue>, dep_events: Vec<EventPtr>) -> UniquePtr<Event>;
        fn wait(queue: &mut UniquePtr<Queue>);

        unsafe fn launch_1d(
            queue: &mut UniquePtr<Queue>,
            global_size: Range1,
            local_size: Range1,
            kernel: &Kernel,
            args: &[&[u8]],
        ) -> UniquePtr<Event>;

        unsafe fn launch_2d(
            queue: &mut UniquePtr<Queue>,
            global_size: Range2,
            local_size: Range2,
            kernel: &Kernel,
            args: &[&[u8]],
        ) -> UniquePtr<Event>;

        unsafe fn launch_3d(
            queue: &mut UniquePtr<Queue>,
            global_size: Range3,
            local_size: Range3,
            kernel: &Kernel,
            args: &[&[u8]],
        ) -> UniquePtr<Event>;
    }
}

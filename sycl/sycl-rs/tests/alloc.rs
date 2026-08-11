//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs::queue::Queue;

#[test]
fn shared_allocation_host_values() {
    let queue = Queue::new();
    let mut array = unsafe { queue.alloc_uninit_shared::<u32>(10) };

    for (index, value) in array.iter_mut().enumerate() {
        *value = index as u32;
    }

    assert_eq!(array.as_ref(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

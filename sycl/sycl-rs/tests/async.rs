//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs::queue::Queue;

#[tokio::test]
async fn check_for_select_support() -> sycl_rs::Result<()> {
    let mut queue = Queue::new();
    let _selected_buffer = tokio::select! {
        buffer1 = queue.alloc_device::<f32>(1024)? => buffer1,
        buffer2 = queue.alloc_device::<f32>(10240)? => buffer2
    };

    Ok(())
}

//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs::queue::Queue;

fn main() {
    let queue = Queue::new();
    let mut buffer = unsafe { queue.alloc_uninit_shared::<u32>(10) };

    for i in 0..buffer.len() {
        buffer[i] = i as u32;
    }

    for e in buffer.iter() { 
        print!("{e} ")
    }

    println!();
}

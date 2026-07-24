//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs::{
    buffer::Buffer,
    kernel::{KernelArgument, KernelArgumentList},
    queue::Queue,
    range::NdRange,
    usm::{SharedAllocator, UsmAllocator},
};

static IOTA_SRC: &str = r#"
#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float start, float *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<float>(id);
}
"#;

#[derive(KernelArgumentList)]
struct IotaArgs<'a> {
    start: f32,
    buffer: &'a mut Buffer<f32, UsmAllocator<SharedAllocator>>,
}

fn main() {
    let mut queue = Queue::new();
    let mut buffer = queue.alloc_shared::<f32>(1024).wait();

    let kernel = queue
        .get_context()
        .create_kernel_bundle_from_source(IOTA_SRC)
        .build()
        .get_kernel("iota");

    unsafe {
        queue.launch(
            NdRange::new([1024], [16]),
            &kernel,
            IotaArgs {
                start: 3.14,
                buffer: &mut buffer,
            },
        )
    }
    .wait();

    for e in buffer.iter() {
        print!("{e} ");
    }
    println!();
}

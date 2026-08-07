//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs::prelude::*;

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

#[tokio::main]
async fn main() {
    let mut queue = Queue::new();
    let mut device_buffer = queue.alloc_device::<f32>(1024).await;

    let kernel = queue
        .context()
        .create_kernel_bundle_from_source(IOTA_SRC)
        .build()
        .kernel("iota");

    unsafe {
        queue.launch(
            NdRange::new([1024], [16]),
            &kernel,
            (3.14_f32, &mut device_buffer),
        )
    }
    .await;

    let mut host_buffer = queue.alloc_host::<f32>(1024).await;

    queue.copy(&device_buffer, &mut host_buffer).await;

    for e in host_buffer.iter() {
        print!("{e} ");
    }
    println!();
}

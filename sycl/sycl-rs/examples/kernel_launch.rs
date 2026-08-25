//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use half::f16;
use sycl_rs::prelude::*;

static IOTA_SRC: &str = r#"
#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(sycl::half start, sycl::half *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<sycl::half>(id);
}
"#;

#[tokio::main]
async fn main() -> sycl_rs::Result<()> {
    let mut queue = Queue::new();
    let mut device_array = queue.alloc_device::<f16>(1024)?.await?;

    let kernel = queue
        .get_context()
        .create_kernel_bundle_from_source(IOTA_SRC)?
        .build()?
        .get_kernel("iota")?;

    unsafe {
        queue.launch(
            NdRange::new([1024], [16]),
            &kernel,
            (f16::from_f32(3.14), &mut device_array),
        )
    }?
    .await?;

    let mut host_array = queue.alloc_host::<f16>(1024)?.await?;

    queue.copy(&device_array, &mut host_array)?.await?;

    for e in host_array.iter() {
        print!("{e} ");
    }
    println!();

    Ok(())
}

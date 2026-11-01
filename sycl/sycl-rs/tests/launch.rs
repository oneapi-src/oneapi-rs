//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs::prelude::*;

static SAXPY_SRC: &str = r#"
#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void saxpy(float a, const float *x, float *y) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    y[id] = a * x[id] + y[id];
}
"#;

// Mixes a plain value with USM pointers, which some backends (such as the OpenCL CPU device)
// only accept when the pointers are passed through their own entry point.
#[test]
fn launch_with_value_and_pointer_arguments() -> sycl_rs::Result<()> {
    const LEN: usize = 64;
    // Queue::new() picks a GPU when there is one. Prefer the CPU, whose OpenCL runtime rejects
    // pointers passed as plain bytes.
    let mut queue = Device::get_devices()
        .iter()
        .find(|device| device.get_info::<info::device::DeviceType>() == info::DeviceType::Cpu)
        .map_or_else(Queue::new, Queue::from);
    let mut x = queue.alloc_shared::<f32>(LEN)?.wait()?;
    let mut y = queue.alloc_shared::<f32>(LEN)?.wait()?;
    for (index, (x, y)) in x.iter_mut().zip(y.iter_mut()).enumerate() {
        *x = index as f32;
        *y = 1.0;
    }

    let kernel = queue
        .get_context()
        .create_kernel_bundle_from_source(SAXPY_SRC)?
        .build()?
        .get_kernel("saxpy")?;
    unsafe {
        queue.launch(
            NdRange::new([LEN as u64], [16]),
            &kernel,
            (2.0f32, &x, &mut y),
        )
    }?
    .wait()?;

    let expected: Vec<f32> = (0..LEN).map(|index| 2.0 * index as f32 + 1.0).collect();
    assert_eq!(y.as_ref(), expected.as_slice());
    Ok(())
}

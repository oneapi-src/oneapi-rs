//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <sycl/sycl.hpp>

namespace sycl_shims {
using Device = sycl::device;
using Platform = sycl::platform;
using Queue = sycl::queue;
using Event = sycl::event;
using Context = sycl::context;
using Kernel = sycl::kernel;
using SourceKernelBundle =
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using ExecutableKernelBundle =
    sycl::kernel_bundle<sycl::bundle_state::executable>;
} // namespace sycl_shims

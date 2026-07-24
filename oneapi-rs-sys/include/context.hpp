//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "oneapi-rs-sys/include/types.hpp"
#include "rust/cxx.h"

namespace sycl_shims {
struct DevicePtr;
} // namespace sycl_shims

namespace sycl_shims::context {
std::unique_ptr<Context> new_context(rust::Vec<DevicePtr>);
} // namespace sycl_shims::context

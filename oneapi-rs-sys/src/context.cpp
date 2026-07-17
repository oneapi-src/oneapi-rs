//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/context.hpp"
#include "oneapi-rs-sys/src/context-sys.rs.h"

namespace sycl_shims::context {
std::unique_ptr<Context> new_context(rust::Vec<DevicePtr> devices) {
  return std::make_unique<Context>();
}
} // namespace sycl_shims::context

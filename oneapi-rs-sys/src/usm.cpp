//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/usm.hpp"
#include "oneapi-rs-sys/src/usm-sys.rs.h"

namespace sycl_shims::usm {
std::uint8_t* aligned_alloc_device(std::size_t alignment, std::size_t num_bytes, Queue const & queue) {
  return static_cast<std::uint8_t*>(sycl::aligned_alloc_device(alignment, num_bytes, queue));
}

std::uint8_t* aligned_alloc_host(std::size_t alignment, std::size_t num_bytes, Queue const & queue) {
  return static_cast<std::uint8_t*>(sycl::aligned_alloc_host(alignment, num_bytes, queue));
}

std::uint8_t* aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes, Queue const & queue) {
  return static_cast<std::uint8_t*>(sycl::aligned_alloc_shared(alignment, num_bytes, queue));
}

void free(std::uint8_t* memory, Queue const & queue) {
  sycl::free(static_cast<void*>(memory), queue);
}
} // namespace sycl_shims::usm

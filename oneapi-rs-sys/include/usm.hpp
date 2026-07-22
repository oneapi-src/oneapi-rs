//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>

#include "oneapi-rs-sys/include/types.hpp"
#include "rust/cxx.h"

namespace sycl_shims::usm {
std::uint8_t *aligned_alloc_device(std::size_t alignment, std::size_t num_bytes,
                                   Queue const &);
std::uint8_t *aligned_alloc_host(std::size_t alignment, std::size_t num_bytes,
                                 Queue const &);
std::uint8_t *aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes,
                                   Queue const &);
void free(std::uint8_t *, Queue const &);
} // namespace sycl_shims::usm

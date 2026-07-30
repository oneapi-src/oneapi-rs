//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "oneapi-rs-sys/include/types.hpp"
#include "rust/cxx.h"

namespace sycl_shims::utils {
template <typename T>
using UnwrappedPtr = std::remove_reference_t<decltype(*std::declval<T>().ptr)>;

template <typename T>
std::vector<UnwrappedPtr<T>> vec_to_vector(rust::Vec<T> &&vec) {
  std::vector<UnwrappedPtr<T>> vector;
  for (auto &&e : vec)
    vector.push_back(std::move(*e.ptr));

  return vector;
}
} // namespace sycl_shims::utils

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

#include "rust/cxx.h"
#include "sycl-rs-sys/include/types.hpp"

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

template <typename T>
rust::Vec<T> vector_to_vec(std::vector<UnwrappedPtr<T>> &&vector) {
  rust::Vec<T> vec;
  for (auto &&e : vector)
    vec.push_back(T{std::make_unique<UnwrappedPtr<T>>(e)});

  return vec;
}
} // namespace sycl_shims::utils

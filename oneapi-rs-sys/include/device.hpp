//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include "rust/cxx.h"

#include <sycl/sycl.hpp>

#include <memory>
#include <vector>

namespace sycl_shims {
struct DevicePtr;
enum class DeviceType: std::uint8_t;

class Device {
public:
  Device(sycl::device p) : inner(p) {}

  static rust::Vec<DevicePtr> get_devices();
  DeviceType get_device_type() const;

private:
  sycl::device inner;
};
} // namespace sycl_shims

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

namespace sycl_shims {
struct DevicePtr;
struct PlatformPtr;
enum class DeviceType : std::uint8_t;
} // namespace sycl_shims

namespace sycl_shims::device {
rust::Vec<DevicePtr> get_devices();
DeviceType get_device_type(Device const &);
rust::String get_version(Device const &);
rust::String get_name(Device const &);
std::unique_ptr<Platform> get_platform(Device const &);
} // namespace sycl_shims::device

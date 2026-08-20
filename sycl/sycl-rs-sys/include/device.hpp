//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>

#include "rust/cxx.h"
#include "sycl-rs-sys/include/types.hpp"

namespace sycl_shims {
struct DevicePtr;
struct PlatformPtr;
enum class DeviceType : std::uint8_t;
enum class Aspect : std::uint8_t;
} // namespace sycl_shims

namespace sycl_shims::device {
rust::Vec<DevicePtr> get_devices();
DeviceType get_device_type(Device const &);
rust::String get_version(Device const &);
rust::String get_name(Device const &);
rust::String get_pci_bdf_address(Device const &);
std::unique_ptr<Platform> get_platform(Device const &);

std::unique_ptr<Device> clone(Device const &);
bool has(Device const &, Aspect);
} // namespace sycl_shims::device

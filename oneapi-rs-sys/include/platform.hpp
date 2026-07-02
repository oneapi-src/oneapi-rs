//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include "rust/cxx.h"
#include "oneapi-rs-sys/include/opaque.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <vector>

namespace sycl_shims::device {
struct DevicePtr;
} // namespace sycl_shims::device

namespace sycl_shims::platform {
struct PlatformPtr;

rust::Vec<PlatformPtr> get_platforms();
rust::Vec<device::DevicePtr> get_devices(Platform const&);
rust::String get_version(Platform const&);
rust::String get_name(Platform const&);
rust::String get_vendor(Platform const&);
} // namespace sycl_shims::platform

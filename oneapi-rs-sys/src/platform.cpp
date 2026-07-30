//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/platform.hpp"
#include "oneapi-rs-sys/include/utils.hpp"
#include "oneapi-rs-sys/src/platform-sys.rs.h"

using sycl_shims::utils::vector_to_vec;

namespace sycl_shims::platform {
rust::Vec<PlatformPtr> get_platforms() {
  return vector_to_vec<PlatformPtr>(sycl::platform::get_platforms());
}

rust::Vec<DevicePtr> get_devices(Platform const &platform) {
  return vector_to_vec<DevicePtr>(platform.get_devices());
}

rust::String get_version(Platform const &platform) {
  return platform.get_info<sycl::info::platform::version>();
}

rust::String get_name(Platform const &platform) {
  return platform.get_info<sycl::info::platform::name>();
}

rust::String get_vendor(Platform const &platform) {
  return platform.get_info<sycl::info::platform::vendor>();
}
} // namespace sycl_shims::platform

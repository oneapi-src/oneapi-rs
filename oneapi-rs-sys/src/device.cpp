//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/device.hpp"
#include "oneapi-rs-sys/src/device-sys.rs.h"

using dt = sycl::info::device_type;

namespace sycl_shims {
rust::Vec<DevicePtr> Device::get_devices() {
  rust::Vec<DevicePtr> devices;

  for (auto &&device : sycl::device::get_devices())
    devices.push_back(DevicePtr { std::make_unique<Device>(device) });

  return devices;
}

DeviceType Device::get_device_type() const {
  auto type = this->inner.get_info<sycl::info::device::device_type>();

  switch(type) {
    case dt::cpu:
      return DeviceType::Cpu;
    case dt::gpu:
      return DeviceType::Gpu;
    case dt::accelerator:
      return DeviceType::Accelerator;
    case dt::custom:
      return DeviceType::Custom;
    case dt::automatic:
      return DeviceType::Automatic;
    case dt::all:
      return DeviceType::All;
    default:
      return DeviceType::Unimplemented;
  }
}

rust::String Device::get_version() const {
  return this->inner.get_info<sycl::info::device::version>();
}
} // namespace sycl_shims

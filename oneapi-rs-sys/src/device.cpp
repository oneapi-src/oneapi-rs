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

namespace sycl_shims::device {
rust::Vec<DevicePtr> get_devices() {
  rust::Vec<DevicePtr> devices;

  for (auto &&device : sycl::device::get_devices())
    devices.push_back(DevicePtr{std::make_unique<Device>(device)});

  return devices;
}

DeviceType get_device_type(Device const &device) {
  auto type = device.get_info<sycl::info::device::device_type>();

  switch (type) {
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

rust::String get_version(Device const &device) {
  return device.get_info<sycl::info::device::version>();
}

rust::String get_name(Device const &device) {
  return device.get_info<sycl::info::device::name>();
}

std::unique_ptr<Platform> get_platform(Device const &device) {
  return std::make_unique<Platform>(device.get_platform());
}

std::unique_ptr<Device> clone(Device const &device) {
  return std::make_unique<Device>(sycl::device(device));
}
} // namespace sycl_shims::device

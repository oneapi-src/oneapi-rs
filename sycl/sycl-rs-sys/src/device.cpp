//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "sycl-rs-sys/include/device.hpp"
#include "sycl-rs-sys/include/utils.hpp"
#include "sycl-rs-sys/src/device-sys.rs.h"

using sycl_shims::utils::vector_to_vec;
using dt = sycl::info::device_type;

namespace syclext = sycl::ext::oneapi;

namespace sycl_shims::device {
rust::Vec<DevicePtr> get_devices() {
  return vector_to_vec<DevicePtr>(sycl::device::get_devices());
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

bool has(Device const &device, Aspect aspect) {
  switch (aspect) {
  case Aspect::ExtIntelPciAddress:
    return device.has(sycl::aspect::ext_intel_pci_address);
  case Aspect::ExtOneapiIsIntegratedGpu:
    return device.has(sycl::aspect::ext_oneapi_is_integrated_gpu);
  }

  return false;
}

rust::String get_version(Device const &device) {
  return device.get_info<sycl::info::device::version>();
}

rust::String get_name(Device const &device) {
  return device.get_info<sycl::info::device::name>();
}

rust::String get_pci_bdf_address(Device const &device) {
  return device.get_info<sycl::ext::intel::info::device::pci_address>();
}

std::unique_ptr<Platform> get_platform(Device const &device) {
  return std::make_unique<Platform>(device.get_platform());
}

std::unique_ptr<Device> clone(Device const &device) {
  return std::make_unique<Device>(sycl::device(device));
}

void enable_peer_access(std::unique_ptr<Device> &device, Device const &peer) {
  return device->ext_oneapi_enable_peer_access(peer);
}

void disable_peer_access(std::unique_ptr<Device> &device, Device const &peer) {
  return device->ext_oneapi_disable_peer_access(peer);
}

bool can_access_peer(std::unique_ptr<Device> &device, Device const &peer,
                     PeerAccess value) {
  syclext::peer_access access;
  switch (value) {
  case PeerAccess::AccessSupported:
    access = syclext::peer_access::access_supported;
  case PeerAccess::AtomicsSupported:
    access = syclext::peer_access::atomics_supported;
  }
  return device->ext_oneapi_can_access_peer(peer, access);
}
} // namespace sycl_shims::device

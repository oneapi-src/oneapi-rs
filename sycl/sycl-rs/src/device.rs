//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs_sys::device::ffi;

pub use sycl_rs_sys::types::ffi::{Aspect, PeerAccess};

use crate::{Result, info::InfoTarget, platform::Platform, private::Sealed};

/// The `Device` struct encapsulates a single SYCL device on which kernels can be executed.
///
/// The `Device` struct provides the common reference semantics.
pub struct Device(pub(crate) cxx::UniquePtr<ffi::Device>);

impl Sealed for Device {}
impl InfoTarget for Device {}

impl From<cxx::UniquePtr<ffi::Device>> for Device {
    fn from(value: cxx::UniquePtr<ffi::Device>) -> Self {
        Self(value)
    }
}

impl Device {
    /// Returns a [`Vec`] containing all the root devices from all SYCL backends
    /// available in the system which have the device type encapsulated by [`DeviceType`](crate::info::DeviceType).
    pub fn get_devices() -> Vec<Self> {
        ffi::get_devices()
            .into_iter()
            .map(|device| Self(device.ptr))
            .collect()
    }

    /// Returns the associated SYCL platform.
    pub fn get_platform(&self) -> Platform {
        let raw_platform = ffi::get_platform(&self.0);
        Platform(raw_platform)
    }

    /// Returns whether this device has the requested aspect.
    pub fn has(&self, aspect: Aspect) -> bool {
        ffi::has(&self.0, aspect)
    }

    /// Queries the peer access status between this device and `peer` according to the query
    /// `value`.
    ///
    /// [`PeerAccess::AccessSupported`]: Returns true only if it is possible for this device to
    /// enable peer access to USM device memory allocations located on the peer device.
    ///
    /// [`PeerAccess::AtomicsSupported`]: When this query returns true, it indicates that this
    /// device may concurrently access and atomically modify USM device memory allocations located
    /// on the peer device when peer access is enabled to that device.
    pub fn can_access_peer(&mut self, peer: &Device, value: PeerAccess) -> bool {
        ffi::can_access_peer(&mut self.0, &peer.0, value)
    }

    /// Enables this device to access USM device allocations located on the peer device.
    /// This does not permit the peer device to access this device’s memory.
    ///
    /// Once this access is enabled, SYCL kernel functions and the explicit memory functions may
    /// access USM device allocations on the peer device subject to the normal rules about context
    /// as described in the core SYCL specification.
    pub fn enable_peer_access(&mut self, peer: &Device) -> Result<()> {
        ffi::enable_peer_access(&mut self.0, &peer.0)
    }

    /// Disables access to the peer device’s memory from this device.
    pub fn disable_peer_access(&mut self, peer: &Device) -> Result<()> {
        ffi::disable_peer_access(&mut self.0, &peer.0)
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        ffi::clone(&self.0).into()
    }
}

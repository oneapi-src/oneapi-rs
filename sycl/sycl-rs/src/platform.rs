//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs_sys::platform::ffi;

use crate::{device::Device, info::InfoTarget, private::Sealed};

/// Abstraction for SYCL platform.
///
/// The `Platform` struct encapsulates a single SYCL platform on which SYCL kernel functions may be executed.
/// A SYCL platform must be associated with a single SYCL backend.
///
/// A `Platform` is also associated with one or more SYCL devices associated with the same SYCL backend.
pub struct Platform(pub(crate) cxx::UniquePtr<ffi::Platform>);

impl Sealed for Platform {}
impl InfoTarget for Platform {}

impl Platform {
    /// Returns a [`Vec`] containing all SYCL platforms from all SYCL backends available in the system.
    pub fn get_platforms() -> Vec<Self> {
        ffi::get_platforms()
            .into_iter()
            .map(|platform| Self(platform.ptr))
            .collect()
    }

    /// Returns a [`Vec`] containing all the root devices associated with this `Platform`.
    pub fn get_devices(&self) -> Vec<Device> {
        ffi::get_devices(&self.0)
            .into_iter()
            .map(|device| Device(device.ptr))
            .collect()
    }
}

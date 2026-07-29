//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#[path = "./info/platform-info.rs"]
pub mod platform;

#[path = "./info/device-info.rs"]
pub mod device;

#[path = "./info/event-info.rs"]
pub mod event;

/// The type of the SYCL device.
pub use oneapi_rs_sys::device::ffi::DeviceType;

/// Event status of the contained action associated with this event.
pub use oneapi_rs_sys::event::ffi::EventCommandStatus;

use crate::private::Sealed;

/// Types which can return an Item of information for a given Target.
pub trait Info: Sealed {
    type Item;
    type Target;

    /// Returns information for a given Target.
    fn get_item(target: &Self::Target) -> Self::Item;
}

/// Types which can be queried for information.
pub trait InfoTarget: Sealed {
    /// Queries this object for information requested by given generic parameter.
    fn get_info<T: Info<Target = Self>>(&self) -> T::Item {
        T::get_item(self)
    }
}

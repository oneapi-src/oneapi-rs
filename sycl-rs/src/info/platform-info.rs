//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::info::Info;
use crate::platform::Platform;
use crate::private::Sealed;
use sycl_rs_sys::platform::ffi;

/// Returns a backend-defined platform version.
pub struct Version;
impl Sealed for Version {}
impl Info for Version {
    type Item = String;
    type Target = Platform;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_version(&target.0)
    }
}

/// Returns the name of the platform.
pub struct Name;
impl Sealed for Name {}
impl Info for Name {
    type Item = String;
    type Target = Platform;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_name(&target.0)
    }
}

/// Returns the name of the vendor providing the platform.
pub struct Vendor;
impl Sealed for Vendor {}
impl Info for Vendor {
    type Item = String;
    type Target = Platform;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_vendor(&target.0)
    }
}

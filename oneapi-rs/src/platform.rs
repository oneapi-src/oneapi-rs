//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use oneapi_rs_sys::platform::ffi;

use crate::info::platform::PlatformInfo;

pub struct Platform(pub(crate) cxx::UniquePtr<ffi::Platform>);

impl Platform {
    pub fn get_platforms() -> Vec<Self> {
        ffi::Platform::get_platforms()
            .into_iter()
            .map(|platform| Self(platform.ptr))
            .collect()
    }

    pub fn get_info<T: PlatformInfo>(&self) -> T::Item {
        T::get_item(self)
    }
}

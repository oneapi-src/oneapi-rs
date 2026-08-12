//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

//! # oneAPI-rs
//! oneAPI is an open, cross-industry, standards-based, unified, multiarchitecture, multi-vendor
//! programming model that delivers a common developer experience across accelerator architectures
//! – for faster application performance, more productivity, and greater innovation.
//!
//! This crate provides Rust bindings for oneAPI libraries. For more information check out each
//! module's documentation.
//!
//! # System dependencies
//! Make sure to install the [Intel oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html).
//! Then source the `setvars.sh` file:
//! ```bash
//! source <oneapi_install_directory>/setvars.sh
//! ```
//!
//! This project was tested on oneAPI Toolkit 2026.1 and requires the Unified Runtime over Level Zero
//! driver version 1.14.37020 or newer. For more detailed information check out the
//! [required extensions](crate#required-extensions) section.

#[doc(inline)]
pub use sycl_rs as sycl;

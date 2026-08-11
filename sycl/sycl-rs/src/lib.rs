//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

//! # SYCL-rs
//! SYCL-rs is a set of (mostly) safe Rust bindings for SYCL - an open, royalty-free,
//! cross-platform abstraction layer that enables code for heterogeneous and offload processors to
//! be written using modern ISO C++, and provides APIs and abstractions to find devices
//! (CPUs, GPUs, FPGAs ...) on which code can be executed, and to manage data resources and code
//! execution on those devices.
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
//!
//! # Getting started
//! ### Building the crate
//! Before building this crate you need to source the `setvars.sh` file. You can then build it as
//! usual with cargo:
//! ```bash
//! cargo build --release
//! ```
//!
//! You must also source `setvars.sh` before running any SYCL program.
//!
//! ### Hello world
//!
//! ```
//! # use sycl_rs::prelude::*;
//!
//! # static IOTA_SRC: &str = r#"
//! # #include <sycl/sycl.hpp>
//! # namespace syclext = sycl::ext::oneapi;
//! # namespace syclexp = sycl::ext::oneapi::experimental;
//! #
//! # extern "C"
//! # SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
//! # void iota(float start, float *ptr) {
//! #     size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
//! #     ptr[id] = start + static_cast<float>(id);
//! # }
//! # "#;
//! #
//! fn main() -> sycl_rs::Result<()> {
//!     // 1. Create a Queue. It's the main entry point to the SYCL API.
//!     let mut queue = Queue::new();
//!     let mut device_array = queue.alloc_device::<f32>(1024)?.wait()?;
//!
//!     // 3. Build a SYCL kernel.
//!     let kernel = queue
//!         .get_context()
//!         .create_kernel_bundle_from_source(IOTA_SRC)?
//!         .build()?
//!         .get_kernel("iota")?;
//!
//!     // 4. Launch your kernel.
//!     unsafe {
//!         queue.launch(
//!             NdRange::new([1024], [16]),
//!             &kernel,
//!             (3.14_f32, &mut device_array),
//!         )
//!     }?
//!     .wait()?;
//!
//!     let mut host_array = queue.alloc_host::<f32>(1024)?.wait()?;
//!
//!     // 5. Copy your data to the host.
//!     queue.copy(&device_array, &mut host_array)?.wait()?;
//!
//!     // You can access your host data just like a normal Rust slice.
//!     for e in host_array.iter() {
//!         print!("{e} ");
//!     }
//!     println!();
//!
//!     Ok(())
//! }
//! ```
//!
//! # Safety model
//! - USM allocations are represented by a zero-cost [`UsmBox`](crate::usmbox::UsmBox) type managed
//!   through RAII.
//!   - Note: `UsmBox` arrays do not rely on accessors, unlike SYCL buffers.
//! - `UsmBox`es are zero-initialized by default.
//! - `UsmBox`es can only store types that implement [`bytemuck::Pod`].
//! - Kernel launch is inherently unsafe. In particular, the caller must ensure that every argument
//!   has the correct representation, layout, and alignment.
//!
//! # Asynchronous programming model
//! Each queue operation returns an [`Event`](`crate::event::Event`). You can synchronously
//! [`.wait()`](crate::event::Event::wait) for it, or asynchronously `.await` it.
//!
//! You can also synchronously call [`Queue::wait()`](crate::queue::Queue::wait) to wait for a
//! [`Queue`](crate::queue::Queue) directly. To do the same asynchronously you have to `.await` an
//! event returned by [`Queue::barrier()`](crate::queue::Queue::barrier).
//!
//! All basic SYCL wrapper types (`Queue`, `Event`, `Context`, `Platform`, `Device`) are thread safe as
//! indicated by the provided [`Send`] and [`Sync`] trait implementations. However - `UsmBox`es are
//! not thread-safe. If you need a thread-safe `UsmBox` you need to wrap it in an `Arc<Mutex<T>>`.
//!
//! # Required extensions
//! This project requires the following SYCL extensions to work:
//! - [sycl_ext_oneapi_kernel_compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)
//! - [sycl_ext_oneapi_raw_kernel_arg](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_raw_kernel_arg.asciidoc)
//!
//! The following extensions are also required for async support:
//! - [sycl_ext_intel_queue_immediate_command_list](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc)
//! - [sycl_ext_oneapi_enqueue_barrier](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc)

pub mod context;
pub mod device;
pub mod event;
pub mod info;
pub mod kernel;
pub mod platform;
pub mod prelude;
pub mod queue;
pub mod range;
pub mod usm;
pub mod usmbox;

pub type SyclError = cxx::Exception;
pub type Result<T> = std::result::Result<T, SyclError>;

mod private {
    pub trait Sealed {}
}

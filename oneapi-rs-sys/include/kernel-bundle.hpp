//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "oneapi-rs-sys/include/types.hpp"
#include "rust/cxx.h"

namespace sycl_shims::kernel_bundle {
std::unique_ptr<SourceKernelBundle>
create_kernel_bundle_from_source(Context const &ctxt, rust::Str source);
std::unique_ptr<ExecutableKernelBundle>
build(std::unique_ptr<SourceKernelBundle> &source);
std::unique_ptr<Kernel> get_kernel(std::unique_ptr<ExecutableKernelBundle> &,
                                   rust::Str);
} // namespace sycl_shims::kernel_bundle

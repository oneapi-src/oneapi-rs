//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/kernel-bundle.hpp"
#include "oneapi-rs-sys/src/kernel-bundle-sys.rs.h"

namespace syclexp = sycl::ext::oneapi::experimental;

namespace sycl_shims::kernel_bundle {
std::unique_ptr<SourceKernelBundle>
create_kernel_bundle_from_source(Context const &ctxt, rust::Str source) {
  return std::make_unique<SourceKernelBundle>(
      syclexp::create_kernel_bundle_from_source(
          ctxt, syclexp::source_language::sycl, std::string(source)));
}

std::unique_ptr<ExecutableKernelBundle>
build(std::unique_ptr<SourceKernelBundle> &source) {
  return std::make_unique<ExecutableKernelBundle>(syclexp::build(*source));
}

std::unique_ptr<Kernel>
get_kernel(std::unique_ptr<ExecutableKernelBundle> &bundle, rust::Str name) {
  return std::make_unique<Kernel>(
      bundle->ext_oneapi_get_kernel(static_cast<std::string const &>(name)));
}
} // namespace sycl_shims::kernel_bundle

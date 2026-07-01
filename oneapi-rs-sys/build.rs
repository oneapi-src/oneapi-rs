//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

fn main() {
    let compiler_root = std::env::var("CMPLR_ROOT")
        .expect("No valid OneAPI installation found.");

    let rust_sources = [
        "src/platform-sys.rs"
    ];

    let cpp_sources = [
        "src/platform.cpp"
    ];

    let cpp_headers = [
        "include/platform.hpp"
    ];

    cxx_build::bridges(&rust_sources)
        .compiler(format!("{compiler_root}/bin/icpx"))
        .include(format!("{compiler_root}/include"))
        .flag("-fsycl")
        .files(&cpp_sources)
        .std("c++17")
        .compile("oneapi-shim");

    println!("cargo::rustc-link-lib=sycl");
    println!("cargo::rustc-link-lib=ze_loader");
    println!("cargo::rustc-link-lib=intlc");

    for source in cpp_sources {
        println!("cargo::rerun-if-changed={source}");
    }

    for header in cpp_headers {
        println!("cargo::rerun-if-changed={header}");
    }
}

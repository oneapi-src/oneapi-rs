//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use std::{io::Error, path::PathBuf};
use which::which;

fn main() {
    let compiler_path = get_compiler_path()
        .expect("Expecting a compiler. Set the ONEAPI_CXX environment variable.");

    let rust_sources = [
        "src/types-sys.rs",
        "src/platform-sys.rs",
        "src/device-sys.rs",
        "src/queue-sys.rs",
        "src/usm-sys.rs",
    ];

    let cpp_sources = [
        "src/platform.cpp",
        "src/device.cpp",
        "src/queue.cpp",
        "src/usm.cpp",
    ];

    let cpp_headers = [
        "include/types.hpp",
        "include/platform.hpp",
        "include/device.hpp",
        "include/queue.hpp",
        "include/usm.hpp",
    ];

    cxx_build::bridges(&rust_sources)
        .compiler(&compiler_path)
        .flag("-fsycl")
        .files(&cpp_sources)
        .std("c++17")
        .compile("oneapi-shim");

    println!("cargo::rustc-link-lib=sycl");
    println!("cargo::rustc-link-lib=intlc");

    for source in cpp_sources {
        println!("cargo::rerun-if-changed={source}");
    }

    for header in cpp_headers {
        println!("cargo::rerun-if-changed={header}");
    }
}

fn get_compiler_path() -> Result<PathBuf, Error> {
    if let Ok(path) = std::env::var("ONEAPI_CXX") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }
    if let Ok(path) = std::env::var("CMPLR_ROOT") {
        let path = PathBuf::from(path).join("bin/icpx");
        if path.exists() {
            return Ok(path)
        }
    }
    if let Ok(path) = which("icpx") {
        return Ok(path);
    }
    if let Ok(path) = which("dpcpp") {
        return Ok(path);
    }
    if let Ok(path) = which("clang++") {
        return Ok(path);
    }

    Err(Error::new(std::io::ErrorKind::NotFound, "No DPC++ compiler found"))
}

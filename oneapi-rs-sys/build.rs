fn main() {
    let compiler_root = std::env::var("CMPLR_ROOT")
        .expect("No valid OneAPI installation found.");

    cxx_build::bridge("src/main.rs")
        .compiler(format!("{compiler_root}/bin/icpx"))
        .include(format!("{compiler_root}/include"))
        .flag("-fsycl")
        .file("src/shim.cpp")
        .std("c++17")
        .compile("oneapi-shim");

    println!("cargo::rustc-link-lib=sycl");
    println!("cargo::rerun-if-changed=include/shim.hpp");
    println!("cargo::rerun-if-changed=include/shim.cpp");
}

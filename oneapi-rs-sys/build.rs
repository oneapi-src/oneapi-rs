fn main() {
    cxx_build::bridge("src/main.rs")
        .include("include")
        .file("src/shim.cpp")
        .std("c++17")
        .compile("test");

    println!("cargo:rerun-if-changed=src/test.cpp");
    println!("cargo:rerun-if-changed=include/shim.hpp");
}
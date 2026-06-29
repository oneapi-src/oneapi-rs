#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("oneapi-rs-sys/include/shim.hpp");
        fn test() -> i32;
    }
}

fn main() {
    println!("Calling: {}", ffi::test());
}

# SYCL-rs

Rust bindings for SYCL and the Intel oneAPI programming environment.

> [!WARNING]
> SYCL-rs is experimental, has not released version 0.1.0, and is not ready
> for production use. Its API may change without notice.

## Supported platforms

The tested configuration is:

- Linux
- Intel oneAPI Toolkit 2026.1
- A SYCL device supported by the installed oneAPI runtime

Other operating systems, oneAPI releases, and SYCL implementations are not
currently tested or supported.

## Prerequisites

- A Rust toolchain with Rust 2024 edition support
- Intel oneAPI Toolkit 2026.1, including the DPC++/C++ compiler and SYCL runtime

Initialize the oneAPI environment before building or running the project:

```bash
source /opt/intel/oneapi/setvars.sh
```

Replace `/opt/intel/oneapi` if the toolkit is installed elsewhere. The build
also accepts an explicit compiler through `ONEAPI_CXX`; the compiler must
support C++17 and SYCL.

## Installation

Add your required dependencies with Cargo:

```bash
cargo add sycl-rs
```

Make sure to source the `setvars.sh` file before building your project.

```bash
source /opt/intel/oneapi/setvars.sh
cargo build
```

## Quick start

Build the workspace from source:
```bash
git clone https://github.com/oneapi-src/oneapi-rs.git
cd oneapi-rs
source /opt/intel/oneapi/setvars.sh
cargo build --workspace
```

List the SYCL devices visible to the runtime:

```bash
source /opt/intel/oneapi/setvars.sh
cargo run -p sycl-rs --example sycl-ls
```

The kernel launch examples demonstrate allocation, runtime kernel compilation,
kernel submission, and copying results back to the host:

```bash
cargo run -p sycl-rs --example kernel_launch
```

## Documentation

- [API documentation](https://docs.rs/sycl_rs/)
- [Examples](sycl/sycl-rs/examples)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

Licensed under either the [Apache License 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

# Contributing to oneAPI-rs

## Development setup

Install a Rust toolchain with Rust 2024 edition support and Intel oneAPI Toolkit
2026.1. Clone the repository and initialize the oneAPI environment:

```bash
git clone https://github.com/oneapi-src/oneapi-rs.git
cd oneapi-rs
source /opt/intel/oneapi/setvars.sh
cargo build --workspace
```

Replace `/opt/intel/oneapi` if the toolkit is installed elsewhere. To select a
specific SYCL compiler, set `ONEAPI_CXX` to its full path.

## Formatting

Check Rust formatting:

```bash
cargo fmt --all -- --check
```

Check tracked C and C++ sources with `clang-format`:

```bash
git ls-files -z -- ':(glob)**/*.cpp' ':(glob)**/*.hpp' ':(glob)**/*.h' \
    | xargs -0 --no-run-if-empty clang-format --dry-run --Werror
```

To apply Rust formatting, run `cargo fmt --all`. Apply C++ formatting with the
project's installed `clang-format` before rerunning the check above.

## Tests and examples

Initialize the oneAPI environment in the current shell, then run the workspace
tests:

```bash
source /opt/intel/oneapi/setvars.sh
cargo test --workspace --verbose
```

The tests and examples require a working SYCL runtime and visible compatible
device.

## License

oneAPI-rs is licensed under either the [Apache License 2.0](LICENSE-APACHE) or
the [MIT License](LICENSE-MIT), at your option. By contributing to the project,
you agree to the license and copyright terms therein and release your contribution
under these terms.

## Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies
that you wrote the patch or otherwise have the right to pass it on as an open-source patch.
The rules are pretty simple: if you can certify the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

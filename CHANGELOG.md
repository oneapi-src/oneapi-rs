# Changelog

All notable changes to this project will be documented in this file.

## Semantic versioning policy

SYCL-rs follows Semantic Versioning. Before 1.0.0, the public API is
experimental: minor releases may contain breaking API changes, while patch
releases are intended for compatible fixes.

## Unreleased

This is the first release of SYCL-rs, a set of (mostly) safe Rust bindings
for SYCL - an open, royalty-free, cross-platform abstraction layer that
enables code for heterogeneous and offload processors to be written using
modern ISO C++, and provides APIs and abstractions to find devices (CPUs,
GPUs, FPGAs …) on which code can be executed, and to manage data resources
and code execution on those devices.

Documentation of SYCL-rs including API description, architecture, and examples
can be found at: <https://docs.rs/sycl_rs/>

### Added

- Initial Rust API for discovering SYCL platforms and devices.
- Queues, events, contexts, USM allocations, and host/device memory operations.
- Runtime kernel bundle compilation and one-, two-, and three-dimensional
  kernel launch support.
- Derive support for typed kernel argument lists.

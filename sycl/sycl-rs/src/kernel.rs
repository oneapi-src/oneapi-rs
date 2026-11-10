//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::Result;
use bytemuck::Pod;
use sycl_rs_sys::{kernel_bundle::ffi, types};

/// A kernel bundle which stores loaded SYCL source code.
pub struct SourceKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::SourceKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::SourceKernelBundle>> for SourceKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::SourceKernelBundle>) -> Self {
        Self(value)
    }
}

impl SourceKernelBundle {
    pub fn build(&mut self) -> Result<ExecutableKernelBundle> {
        ffi::build(&mut self.0).map(Into::into)
    }
}

/// A kernel bundle which stores compiled SYCL kernels.
pub struct ExecutableKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::ExecutableKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::ExecutableKernelBundle>> for ExecutableKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::ExecutableKernelBundle>) -> Self {
        Self(value)
    }
}

impl ExecutableKernelBundle {
    pub fn get_kernel(&mut self, name: &str) -> Result<Kernel> {
        ffi::get_kernel(&mut self.0, name).map(Into::into)
    }
}

/// An executable SYCL kernel.
pub struct Kernel(pub(crate) cxx::UniquePtr<types::ffi::Kernel>);

impl From<cxx::UniquePtr<types::ffi::Kernel>> for Kernel {
    fn from(value: cxx::UniquePtr<types::ffi::Kernel>) -> Self {
        Self(value)
    }
}

/// The raw representation of a SYCL kernel argument.
///
/// A backend may take pointers through a different entry point than plain bytes, and cannot
/// tell the two apart from the bytes alone, so pointers are marked as such.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum RawKernelArg<'a> {
    /// The bytes of a trivially copyable value that is not a pointer.
    Bytes(&'a [u8]),
    /// The bytes of a pointer to USM memory.
    Pointer(&'a [u8]),
}

impl<'a> From<RawKernelArg<'a>> for types::ffi::KernelArg<'a> {
    fn from(arg: RawKernelArg<'a>) -> Self {
        let (kind, bytes) = match arg {
            RawKernelArg::Bytes(bytes) => (types::ffi::KernelArgKind::Bytes, bytes),
            RawKernelArg::Pointer(bytes) => (types::ffi::KernelArgKind::Pointer, bytes),
        };
        Self { kind, bytes }
    }
}

/// Types which can be passed as SYCL kernel arguments.
///
/// # Safety
///
/// A type implementing this trait must mirror the representation and alignment of the
/// corresponding SYCL kernel argument structure, and must return [`RawKernelArg::Pointer`]
/// exactly when it represents a pointer to USM memory.
pub unsafe trait KernelArgument {
    /// Converts self to its raw representation.
    ///
    /// # Safety
    ///
    /// This function returns a reference to raw bytes, marked as a pointer or plain bytes.
    /// These bytes will be passed to FFI functions. The caller must make sure these functions
    /// respect Rust's aliasing rules.
    unsafe fn as_raw_arg(&self) -> RawKernelArg<'_>;
}

unsafe impl<T: Pod> KernelArgument for T {
    unsafe fn as_raw_arg(&self) -> RawKernelArg<'_> {
        RawKernelArg::Bytes(bytemuck::bytes_of(self))
    }
}

/// Types which describe an argument list for a SYCL kernel.
///
/// # Safety
///
/// A type implementing this trait must mirror the representation and alignment of each
/// corresponding SYCL kernel argument inside the returned array, and each entry must be
/// [`RawKernelArg::Pointer`] exactly when it represents a pointer to USM memory.
pub unsafe trait KernelArgumentList<const ARGC: usize> {
    /// Converts each struct member to its raw representation.
    ///
    /// # Safety
    ///
    /// This function returns references to raw bytes, each marked as a pointer or plain bytes.
    /// These bytes will be passed to FFI functions. The caller must make sure these functions
    /// respect Rust's aliasing rules.
    unsafe fn as_raw_arg_list(&self) -> [RawKernelArg<'_>; ARGC];
}

unsafe impl KernelArgumentList<0> for () {
    unsafe fn as_raw_arg_list(&self) -> [RawKernelArg<'_>; 0] {
        []
    }
}

unsafe impl<T: KernelArgument> KernelArgumentList<1> for T {
    unsafe fn as_raw_arg_list(&self) -> [RawKernelArg<'_>; 1] {
        [unsafe { self.as_raw_arg() }]
    }
}

pub use sycl_rs_derive::KernelArgumentList;

use sycl_rs_derive::impl_arg_list_for_tuples;

impl_arg_list_for_tuples! {16}

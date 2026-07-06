use std::{alloc::Layout, ptr::NonNull};
use thiserror::Error;

/// A copy of the unstable [`Allocator`](std::alloc::Allocator) trait.
/// This module will become unnecessary when the [`Allocator`](std::alloc::Allocator) trait gets stabilized.

#[derive(Error, Debug)]
pub enum AllocError {
    #[error("the device used does not support USM allocation")]
    FeatureNotSupported,
    #[error("the device used does not correspond to any SYCL context")]
    Invalid,
    #[error("there's not enough memory to allocate")]
    NoMemory,
    #[error("other error")]
    Other(String)
}

pub unsafe trait Allocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

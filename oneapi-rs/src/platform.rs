use oneapi_rs_sys::platform::ffi;

use crate::info::platform::PlatformInfo;

pub struct Platform(pub(crate) cxx::SharedPtr<ffi::Platform>);

impl Platform {
    pub fn get_platforms() -> Vec<Platform> {
        ffi::Platform::get_platforms()
            .into_iter()
            .map(|platform| Platform(platform.ptr.clone()))
            .collect()
    }

    pub fn get_info<T: PlatformInfo>(&self) -> T::Item {
        T::get_item(self)
    }
}

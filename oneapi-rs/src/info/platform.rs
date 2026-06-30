use crate::platform::Platform;

pub trait PlatformInfo {
    type Item;
    fn get_item(platform: &Platform) -> Self::Item;
}

pub struct Version;
impl PlatformInfo for Version {
    type Item = String;
    fn get_item(platform: &Platform) -> Self::Item {
        platform.0.get_version()
    }
}

pub struct Name;
impl PlatformInfo for Name {
    type Item = String;
    fn get_item(platform: &Platform) -> Self::Item {
        platform.0.get_name()
    }
}

pub struct Vendor;
impl PlatformInfo for Vendor {
    type Item = String;
    fn get_item(platform: &Platform) -> Self::Item {
        platform.0.get_name()
    }
}

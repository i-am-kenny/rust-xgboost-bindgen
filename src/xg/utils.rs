use std::path::Path;
use std::{ffi::CString, os::unix::prelude::OsStrExt};

use super::{XGBoostError, XGBoostResult};

pub fn path_to_cstring<P: AsRef<Path>>(path: P) -> XGBoostResult<CString> {
    CString::new(path.as_ref().as_os_str().as_bytes())
        .map_err(|_| XGBoostError::from_str("failed creating cstring from path"))
}

pub fn str_to_cstring(value: &str) -> XGBoostResult<CString> {
    CString::new(value).map_err(|_| XGBoostError::from_str("failed creating cstring from &str"))
}

pub fn string_to_cstring(value: String) -> XGBoostResult<CString> {
    str_to_cstring(value.as_str())
}

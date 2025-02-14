use std::{error, fmt};

use crate::bindings::XGBGetLastError;

pub type XGBoostResult<T> = Result<T, XGBoostError>;

#[derive(Debug)]
pub struct XGBoostError {
    pub(crate) inner: String,
}

impl error::Error for XGBoostError {}

impl fmt::Display for XGBoostError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.inner)
    }
}

impl XGBoostError {
    pub fn from_return_value(return_value: i32) -> XGBoostResult<()> {
        match return_value {
            0 => Ok(()),
            -1 => Err(XGBoostError {
                inner: get_last_error(),
            }),
            _ => Err(XGBoostError {
                inner: "invalid xgboost return_value".into(),
            }),
        }
    }

    pub fn from_string(inner: String) -> Self {
        Self { inner }
    }

    pub fn from_str(inner: &str) -> Self {
        Self {
            inner: inner.to_string(),
        }
    }
}

fn get_last_error() -> String {
    unsafe {
        let message = XGBGetLastError();
        let message = std::ffi::CStr::from_ptr(message);
        message
            .to_str()
            .unwrap_or("xgboost last_error unavailable")
            .to_string()
    }
}

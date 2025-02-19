pub use array_interface::*;
pub use booster::*;
pub use dmatrix::*;
pub use proxy_dmatrix::*;
pub use error::{XGBoostError, XGBoostResult};

mod array_interface;
mod booster;
mod proxy_dmatrix;
mod dmatrix;
mod error;
mod utils;

#[macro_export]
macro_rules! xgboost_call {
    ($call:expr) => {
        XGBoostError::from_return_value(unsafe { $call })
    };
}

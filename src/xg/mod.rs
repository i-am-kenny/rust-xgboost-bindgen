pub use array_interface::*;
pub use booster::*;
pub use dmatrix::*;
pub use error::{XGBoostError, XGBoostResult};
pub use proxy_dmatrix::*;

mod array_interface;
mod booster;
mod dmatrix;
mod error;
mod proxy_dmatrix;
mod utils;

#[macro_export]
macro_rules! xgboost_call {
    ($call:expr) => {
        XGBoostError::from_return_value(unsafe { $call })
    };
}

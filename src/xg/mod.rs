pub use booster::*;
pub use dmatrix::*;
pub use error::{XGBoostError, XGBoostResult};

mod booster;
mod dmatrix;
mod error;
mod utils;

#[macro_export]
macro_rules! xgboost_call {
    ($call:expr) => {
        XGBoostError::from_return_value(unsafe { $call })
    };
}

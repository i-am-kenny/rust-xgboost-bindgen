use crate::bindings;
use std::sync::Arc;

use super::{XGBoostError, XGBoostResult};

pub struct DMatrix {
    pub(crate) handle: Arc<bindings::DMatrixHandle>,
}

unsafe impl Send for DMatrix {}

impl DMatrix {
    pub fn from_array(data: &[f32], number_of_rows: usize) -> XGBoostResult<Self> {
        let mut handle = std::ptr::null_mut();

        crate::xgboost_call!(bindings::XGDMatrixCreateFromMat(
            data.as_ptr(),
            number_of_rows as bindings::bst_ulong,
            (data.len() / number_of_rows) as bindings::bst_ulong,
            f32::NAN,
            &mut handle,
        ))?;

        Ok(Self {
            handle: handle.into(),
        })
    }

    pub fn from_matrix(
        data: &[&[f32]],
        number_of_rows: usize,
        number_of_columns: usize,
    ) -> XGBoostResult<Self> {
        let mut handle = std::ptr::null_mut();

        crate::xgboost_call!(bindings::XGDMatrixCreateFromMat(
            data.as_ptr() as *const f32,
            number_of_rows as bindings::bst_ulong,
            number_of_columns as bindings::bst_ulong,
            f32::NAN,
            &mut handle,
        ))?;

        Ok(Self {
            handle: handle.into(),
        })
    }

    pub fn from_ptr(
        data: *const f64,
        number_of_rows: usize,
        number_of_columns: usize,
    ) -> XGBoostResult<Self> {
        let mut handle = std::ptr::null_mut();

        crate::xgboost_call!(bindings::XGDMatrixCreateFromMat(
            data as *const f32,
            number_of_rows as bindings::bst_ulong,
            number_of_columns as bindings::bst_ulong,
            f32::NAN,
            &mut handle,
        ))?;

        Ok(Self {
            handle: handle.into(),
        })
    }
}

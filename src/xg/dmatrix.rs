use crate::bindings;

use super::{XGBoostError, XGBoostResult};

pub struct DMatrix {
    pub(crate) handle: bindings::DMatrixHandle,
}

unsafe impl Send for DMatrix {}

impl DMatrix {
    pub fn from_file(path: &str) -> XGBoostResult<Self> {
        let silent = 0;
        let mut handle = std::ptr::null_mut();
        let fname =
            std::ffi::CString::new(path).map_err(|e| XGBoostError::from_str(&e.to_string()))?;

        crate::xgboost_call!(bindings::XGDMatrixCreateFromFile(
            fname.as_ptr(),
            silent,
            &mut handle
        ))?;

        Ok(Self { handle })
    }

    pub fn rows(&self) -> XGBoostResult<u64> {
        let mut num_rows = 0;
        crate::xgboost_call!(bindings::XGDMatrixNumRow(self.handle, &mut num_rows))?;

        Ok(num_rows)
    }

    pub fn columns(&self) -> XGBoostResult<u64> {
        let mut num_cols = 0;
        crate::xgboost_call!(bindings::XGDMatrixNumCol(self.handle, &mut num_cols))?;

        Ok(num_cols)
    }

    pub fn from_array(data: &[f32], number_of_rows: usize) -> XGBoostResult<Self> {
        let mut handle = std::ptr::null_mut();

        crate::xgboost_call!(bindings::XGDMatrixCreateFromMat(
            data.as_ptr(),
            number_of_rows as bindings::bst_ulong,
            (data.len() / number_of_rows) as bindings::bst_ulong,
            f32::NAN,
            &mut handle,
        ))?;

        Ok(Self { handle })
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

        Ok(Self { handle })
    }

    pub fn from_ptr(
        data: *const f32,
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

        Ok(Self { handle })
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        crate::xgboost_call!(bindings::XGDMatrixFree(self.handle)).unwrap();
    }
}

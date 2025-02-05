use std::sync::Arc;
use std::{path::Path, ptr, slice};

use crate::{bindings, xg::XGBoostError, DMatrix};

use super::{utils, XGBoostResult};

pub struct Booster {
    pub(crate) handle: Arc<bindings::BoosterHandle>,
}

unsafe impl Send for Booster {}

impl Booster {
    pub fn create() -> XGBoostResult<Self> {
        let mut handle = ptr::null_mut();

        crate::xgboost_call!(bindings::XGBoosterCreate(ptr::null(), 0, &mut handle))?;

        Ok(Self {
            handle: handle.into(),
        })
    }

    pub fn load_model<P: AsRef<Path>>(path: P) -> XGBoostResult<Self> {
        if !path.as_ref().exists() {
            return Err(XGBoostError::from_string(format!(
                "file not found: {}",
                path.as_ref().display()
            )));
        }

        let path = utils::path_to_cstring(path)?;
        let booster = Self::create()?;

        crate::xgboost_call!(bindings::XGBoosterLoadModel(*booster.handle, path.as_ptr()))?;

        Ok(booster)
    }

    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> XGBoostResult<()> {
        let path = utils::path_to_cstring(path)?;

        crate::xgboost_call!(bindings::XGBoosterSaveModel(*self.handle, path.as_ptr()))?;

        Ok(())
    }

    pub fn set_param_native(&self, name: &str, value: &str) -> XGBoostResult<()> {
        let name = utils::str_to_cstring(name)?;
        let value = utils::str_to_cstring(value)?;

        crate::xgboost_call!(bindings::XGBoosterSetParam(
            *self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))?;

        Ok(())
    }

    pub fn update(&self, dtrain: &DMatrix, iteration: i32) -> XGBoostResult<()> {
        crate::xgboost_call!(bindings::XGBoosterUpdateOneIter(
            *self.handle,
            iteration,
            *dtrain.handle
        ))?;

        Ok(())
    }

    pub fn predict(&self, dmatrix: &DMatrix, options: &[PredictOption]) -> XGBoostResult<Vec<f32>> {
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();

        crate::xgboost_call!(bindings::XGBoosterPredict(
            *self.handle,
            *dmatrix.handle,
            PredictOption::as_mask(options),
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;

        if out_result.is_null() {
            return Err(XGBoostError::from_str("booster predicted return null"));
        }

        let out_result = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };

        Ok(out_result)
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        crate::xgboost_call!(bindings::XGBoosterFree(*self.handle))
            .expect("failed dropping booster");
    }
}

impl Clone for Booster {
    fn clone(&self) -> Self {
        // copy handle?
        // save to buffer
        // load from buffer
        todo!()
    }
}

pub enum PredictOption {
    OutputMargin = 1,
    PredictContribution = 2,
    PredictApproximatedContribution = 4,
    PredictFeatureInteraction = 8,
    PredictAprroximatedFeatureInteraction = 16,
}

impl PredictOption {
    pub fn as_mask(options: &[PredictOption]) -> i32 {
        options.iter().fold(0, |acc, i| {
            acc | {
                let value = match i {
                    Self::OutputMargin => 1,
                    Self::PredictContribution => 2,
                    Self::PredictApproximatedContribution => 4,
                    Self::PredictFeatureInteraction => 8,
                    Self::PredictAprroximatedFeatureInteraction => 16,
                };

                acc | value
            }
        })
    }
}

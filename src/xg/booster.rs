use std::{path::Path, ptr, rc::Rc, slice};

use crate::{bindings, xg::XGBoostError, DMatrix};

use super::{utils, XGBoostResult};

pub struct Booster {
    pub(crate) handle: bindings::BoosterHandle,
}

unsafe impl Send for Booster {}

impl Booster {
    pub fn with_cache(cached_mat: &[bindings::DMatrixHandle]) -> XGBoostResult<Self> {
        let mut handle = ptr::null_mut();

        crate::xgboost_call!(bindings::XGBoosterCreate(
            cached_mat.as_ptr(),
            cached_mat.len() as u64,
            &mut handle
        ))?;

        Ok(Self { handle })
    }

    pub(crate) fn new() -> XGBoostResult<Self> {
        let mut handle = ptr::null_mut();

        crate::xgboost_call!(bindings::XGBoosterCreate(ptr::null(), 0, &mut handle))?;

        Ok(Self { handle })
    }

    pub fn load_model<P: AsRef<Path>>(path: P) -> XGBoostResult<Self> {
        if !path.as_ref().exists() {
            return Err(XGBoostError::from_string(format!(
                "file not found: {}",
                path.as_ref().display()
            )));
        }

        let path = utils::path_to_cstring(path)?;
        let booster = Self::new()?;

        crate::xgboost_call!(bindings::XGBoosterLoadModel(booster.handle, path.as_ptr()))?;

        Ok(booster)
    }

    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> XGBoostResult<()> {
        let path = utils::path_to_cstring(path)?;

        crate::xgboost_call!(bindings::XGBoosterSaveModel(self.handle, path.as_ptr()))?;

        Ok(())
    }

    pub fn set_param(&self, name: &str, value: &str) -> XGBoostResult<()> {
        let name = utils::str_to_cstring(name)?;
        let value = utils::str_to_cstring(value)?;

        crate::xgboost_call!(bindings::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))?;

        Ok(())
    }
    pub fn eval(
        &self,
        eval_dmats: &[&DMatrix],
        eval_names: &[&str],
        iteration: i32,
    ) -> XGBoostResult<()> {
        let mut handles = eval_dmats.iter().map(|d| d.handle).collect::<Vec<_>>();
        let handles = handles.as_mut_ptr();

        let eval_names: Vec<_> = eval_names
            .iter()
            .map(|n| std::ffi::CString::new(*n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| XGBoostError {
                inner: e.to_string(),
            })?;

        let mut eval_names: Vec<_> = eval_names.iter().map(|c| c.as_ptr()).collect();
        let eval_names = eval_names.as_mut_ptr();

        let mut out = std::ptr::null();

        let return_code = unsafe {
            bindings::XGBoosterEvalOneIter(
                self.handle,
                iteration,
                handles,
                eval_names,
                eval_dmats.len() as u64,
                &mut out,
            )
        };

        let out_result = unsafe { std::ffi::CStr::from_ptr(out).to_str().unwrap() };

        tracing::debug!(out_result, "eval_one_iter");

        XGBoostError::from_return_value(return_code)
    }

    pub fn update(&self, dtrain: &DMatrix, iteration: i32) -> XGBoostResult<()> {
        crate::xgboost_call!(bindings::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))?;

        Ok(())
    }

    pub fn num_features(&self) -> XGBoostResult<u64> {
        let mut out = 0;

        let return_value = unsafe { bindings::XGBoosterGetNumFeature(self.handle, &mut out) };

        XGBoostError::from_return_value(return_value)?;

        Ok(out)
    }

    pub fn predict(&self, dmatrix: &DMatrix, options: &[PredictOption]) -> XGBoostResult<Vec<f32>> {
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();

        crate::xgboost_call!(bindings::XGBoosterPredict(
            self.handle,
            dmatrix.handle,
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

    pub fn predict_from_dmatrix(&self, dmatrix: &DMatrix) -> XGBoostResult<(Vec<u64>, Vec<f32>)> {
        let shape: Rc<u64> = Rc::new(0u64);
        let shape = Rc::as_ptr(&shape);
        let shape = shape as *mut *const u64;

        let mut out_dim: u64 = 0;
        let mut out_result = ptr::null();

        let config = include_str!("default_predict_config.json");
        let config = std::ffi::CString::new(config).unwrap();

        crate::xgboost_call!(bindings::XGBoosterPredictFromDMatrix(
            self.handle,
            dmatrix.handle,
            config.as_ptr(),
            shape,
            &mut out_dim,
            &mut out_result
        ))?;

        if out_result.is_null() {
            return Err(XGBoostError::from_str("booster predicted return null"));
        }

        let dimensions = unsafe { slice::from_raw_parts(shape, out_dim as usize) };
        let dimensions: Vec<_> = dimensions.iter().map(|s| unsafe { **s }).collect();

        let length = dimensions
            .iter()
            .cloned()
            .reduce(|acc, i| acc * i)
            .unwrap_or_default();

        let out_result = unsafe { slice::from_raw_parts(out_result, length as usize).to_vec() };

        Ok((dimensions, out_result))
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        crate::xgboost_call!(bindings::XGBoosterFree(self.handle))
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

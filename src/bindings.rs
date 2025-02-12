#![allow(clippy::all)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_matrix() {
        let dmat_train = "xgboost/demo/data/agaricus.txt.train?format=libsvm";

        let dmatrix = crate::DMatrix::from_file(dmat_train).unwrap();

        let num_rows = dmatrix.rows().unwrap();
        assert_eq!(num_rows, 6513);

        let num_cols = dmatrix.columns().unwrap();
        assert_eq!(num_cols, 127);
    }

    #[test]
    fn compute_matrix() {
        let mut handle = std::ptr::null_mut();
        let fname = [1.0];
        let ret_val =
            unsafe { XGDMatrixCreateFromMat(fname.as_ptr(), 1, 1, f32::NAN, &mut handle) };
        assert_eq!(ret_val, 0);

        let mut num_rows = 0;
        let ret_val = unsafe { XGDMatrixNumRow(handle, &mut num_rows) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_rows, 1);

        let mut num_cols = 0;
        let ret_val = unsafe { XGDMatrixNumCol(handle, &mut num_cols) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_cols, 1);

        let ret_val = unsafe { XGDMatrixFree(handle) };
        assert_eq!(ret_val, 0);
    }
}

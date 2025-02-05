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

        let silent = 0;
        let mut handle = std::ptr::null_mut();
        let fname = std::ffi::CString::new(dmat_train).unwrap();
        let ret_val = unsafe { XGDMatrixCreateFromFile(fname.as_ptr(), silent, &mut handle) };
        unsafe {
            let message = XGBGetLastError();
            let message = std::ffi::CStr::from_ptr(message);
            println!("{}", message.to_str().unwrap());
        };
        // println!("{message}");
        assert_eq!(ret_val, 0);

        let mut num_rows = 0;
        let ret_val = unsafe { XGDMatrixNumRow(handle, &mut num_rows) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_rows, 6513);

        let mut num_cols = 0;
        let ret_val = unsafe { XGDMatrixNumCol(handle, &mut num_cols) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_cols, 127);

        let ret_val = unsafe { XGDMatrixFree(handle) };
        assert_eq!(ret_val, 0);
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

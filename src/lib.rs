mod bindings;
mod xg;

pub use xg::*;

#[cfg(test)]
mod tests {

    #[test]
    fn read_matrix() {
        let dmat_train = "xgboost/demo/data/agaricus.txt.train?format=libsvm";

        let dmatrix = crate::DMatrix::from_file(dmat_train).unwrap();

        let num_rows = dmatrix.rows().unwrap();
        assert_eq!(num_rows, 6513);

        let num_cols = dmatrix.columns().unwrap();
        assert_eq!(num_cols, 127);
    }

    /// integration test
    #[test]
    fn predict() {
        tracing_subscriber::fmt::init();

        let dmat_train = "xgboost/demo/data/agaricus.txt.train?format=libsvm";

        let d_train = crate::DMatrix::from_file(dmat_train).unwrap();

        let d_test =
            crate::DMatrix::from_file("xgboost/demo/data/agaricus.txt.test?format=libsvm").unwrap();

        let booster = crate::Booster::with_cache(&[d_train.handle, d_test.handle]).unwrap();

        let parameters = [
            ("device", "cpu"),
            ("objective", "binary:logistic"),
            ("min_child_weight", "1"),
            ("gamma", "0.1"),
            ("max_depth", "3"),
            ("verbosity", "1"),
            ("validate_parameters", "true"),
        ];
        for (key, value) in parameters {
            booster.set_param(key, value).unwrap();
        }

        let eval_names = ["train", "test"];

        let n_trees = 10;
        for i in 0..n_trees {
            booster.update(&d_train, i).unwrap();
            booster.eval(&[&d_train, &d_test], &eval_names, i).unwrap();
        }

        let num_features = booster.num_features().unwrap();

        println!("num_features: {num_features}");

        let ((rows, columns), predictions) = booster.predict_from_dmatrix(&d_test).unwrap();

        let sample = &predictions[..10];

        println!("rows: {rows}, columns: {columns}, predictions: {sample:?}..");

    }
}

use std::{env, path::Path, process::Command};

fn main() {
    let target = env::var("TARGET").expect("TARGET not defined");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not defined");
    let xgb_root = Path::new(&out_dir).join("xgboost");

    if !xgb_root.exists() {
        Command::new("cp")
            .args(["-r", "xgboost", xgb_root.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to copy ./xgboost to {}: {}", xgb_root.display(), e);
            });
    }

    let dst = cmake::Config::new(&xgb_root)
        .define("BUILD_STATIC_LIB", "ON")
        // .define("CC", "gcc-11")
        // .define("CXX", "g++-11")
        .build();
}

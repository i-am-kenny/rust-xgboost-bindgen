use std::{env, path::Path, process::Command};

fn main() {
    let target = env::var("TARGET").expect("TARGET not defined");
    let out_dir = env::var("OUT_DIR")
        .map(std::path::PathBuf::from)
        .expect("OUT_DIR not defined");
    let xgb_root = Path::new(&out_dir).join("xgboost");

    if !xgb_root.exists() {
        Command::new("cp")
            .args(["-r", "xgboost", xgb_root.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to copy ./xgboost to {xgb_root:?}: {e}");
            });
    }

    #[cfg(target_os = "macos")]
    let dst = cmake::Config::new(xgb_root.as_path())
        // only for macos https://github.com/dmlc/xgboost/pull/5397/files
        // .define("BUILD_STATIC_LIB", "ON")
        // .define("CC", "gcc-11")
        // .define("CXX", "g++-11")
        .build();

    // #[cfg(not(target_os = "macos"))]
    let dst = {
        let mut config = cmake::Config::new(xgb_root.as_path());

        config.define("BUILD_STATIC_LIB", "ON");
        #[cfg(feature = "cuda")]
        config.define("USE_CUDA", "ON");

        #[cfg(all(feature = "cuda", feature = "turing"))]
        config.define("CMAKE_CUDA_ARCHITECTURES", "75");

        config
    }
    // .define("CC", "gcc-11")
    // .define("CXX", "g++-11")
    .build();

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .blocklist_item("std::__1.*")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            xgb_root.join("dmlc-core/include").display()
        ))
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-link-search={}", xgb_root.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        xgb_root.join("rabit/lib").display()
    );
    println!(
        "cargo:rustc-link-search={}",
        xgb_root.join("dmlc-core").display()
    );

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=dmlc");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=xgboost");

    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=static=xgboost");

    // #[cfg(feature="cuda")]
    // println!("cargo:rustc-link-lib=cudart");
}

fn main() {

    cc::Build::new()
        .cuda(true)
        .file("src/cuda_main.cu")
        .flag("-std=c++14")
        .flag("-ccbin=/usr/bin/g++-11") 
        .compile("cuda_gravity");

    println!("cargo:rustc-link-lib=static=cuda_gravity");
    println!("cargo:rerun-if-changed=src/cuda_main.cu");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
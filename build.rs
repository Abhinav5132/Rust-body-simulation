fn main() {

    cc::Build::new()
        .cuda(true)
        .flag("-std=c++14")
        .flag("-ccbin")
        .flag("/usr/bin/g++-11")
        .file("src/cuda_main.cu")
        .compile("cuda_gravity");

    println!("cargo:rustc-link-lib=static=cuda_gravity");
    println!("cargo:rerun-if-changed=src/cuda_main.cu");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
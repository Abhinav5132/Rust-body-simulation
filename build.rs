fn main() {
    println!("cargo:rustc-link-search=native=src"); // <--- IMPORTANT: Update this path!
    println!("cargo:rustc-link-lib=dylib=cuda_gravity");
    println!("cargo:rerun-if-changed=src/cuda_main.cu");
    println!("cargo:rustc-link-search=native=/usr/lib/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
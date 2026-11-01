use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=config/mappings.toml");

    for dir in ["assets/models", "assets/fonts", "config", "data"] {
        let p = Path::new(dir);
        if !p.exists() {
            let _ = fs::create_dir_all(p);
        }
    }

    let palm_model = Path::new("assets/models/palm_detection_full.onnx");
    let landmark_model = Path::new("assets/models/hand_landmark_full.onnx");

    if !palm_model.exists() || !landmark_model.exists() {
        println!("cargo:warning=Missing ONNX models in assets/models/. Run scripts/setup_models.ps1 or scripts/setup_models.sh.");
    }
}

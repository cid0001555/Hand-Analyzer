use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GestureEspError {
    #[error("config error at '{path}': {message}")]
    Config { path: PathBuf, message: String },

    #[error("toml parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("file watcher error: {0}")]
    Watcher(#[from] notify::Error),

    #[error("camera error: {0}")]
    Camera(String),

    #[error("model file not found at '{0}'")]
    ModelNotFound(PathBuf),

    #[error("onnx error: {0}")]
    Onnx(#[from] ort::Error),

    #[error("unexpected tensor shape from '{model}': expected {expected}, got {actual:?}")]
    TensorShape {
        model: &'static str,
        expected: &'static str,
        actual: Vec<usize>,
    },

    #[error("classifier error: {0}")]
    Classifier(String),

    #[error("csv error: {0}")]
    DatasetCsv(#[from] csv::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("input error ({backend}): {message}")]
    OsInput {
        backend: &'static str,
        message: String,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type DomainResult<T> = std::result::Result<T, GestureEspError>;

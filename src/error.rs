//! An error when fitting isolation forest algorithm.
use thiserror::Error;

pub type Result<T> = std::result::Result<T, IsolatioinForestError>;

#[derive(Error, Debug)]
pub enum IsolatioinForestError {
    #[error("not enough samples")]
    NotEnoughSamples,
}

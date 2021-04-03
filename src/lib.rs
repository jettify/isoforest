pub mod forest;
pub mod tree;
pub mod error;
pub use forest::{IsolationForest, IsolationForestParams};
pub use tree::{IsolationTree, IsolationTreeParams, MaxFeatures, MaxSamples};

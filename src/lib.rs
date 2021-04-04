pub mod error;
pub mod forest;
pub mod tree;
pub use forest::{IsolationForest, IsolationForestParams};
pub use tree::{IsolationTree, IsolationTreeParams, MaxFeatures, MaxSamples};

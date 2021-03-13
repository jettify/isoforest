# IsolationForest
[![ci-badge](https://github.com/jettify/isoforest/workflows/CI/badge.svg)](https://github.com/jettify/isoforest/actions?query=workflow%3ACI)

 # Example

 ```rust

use isoforest::{IsolationForestParams, IsolationTreeParams};
use ndarray::array;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::io::Result;

use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};

fn main() -> Result<()> {
    let data = array![
        [-2.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -2.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [6.0, 3.0],  // anomaly
        [-4.0, 7.0]  // anomaly
    ];

    let rng = Isaac64Rng::seed_from_u64(4);
    let dataset = DatasetBase::new(data.clone(), ());
    let sample_size = dataset.records().nrows();
    let num_features = dataset.records().ncols();
    let num_trees = 3;

    let tree_params = IsolationTreeParams::new(sample_size, num_features, rng);
    let forest_prams = IsolationForestParams::new(num_trees, &tree_params);
    let model = forest_prams.fit(&dataset).unwrap();

    let preds = model.predict(&data).unwrap();

    println!("{}", preds.mapv(|a| if a > 0.5 { 1 } else { 0 }));
    // expected result [0, 0, 0, 0, 0, 0, 1, 1]
    Ok(())
}
```

 # Lincese
  Licensed under the Apache License, Version 2.0

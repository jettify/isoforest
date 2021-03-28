use isoforest::{IsolationForestParams, IsolationTreeParams, MaxFeatures, MaxSamples};
use ndarray::array;
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

    let dataset = DatasetBase::new(data.clone(), ());
    let tree_params = IsolationTreeParams::default()
        .with_max_samples(MaxSamples::Auto)
        .with_max_features(MaxFeatures::Ratio(1.0))
        .with_seed(3);

    let forest_prams = IsolationForestParams::new(3, &tree_params);

    let model = forest_prams.fit(&dataset).unwrap();
    let preds = model.predict(&data).unwrap();

    println!("{}", preds.mapv(|a| if a > 0.5 { 1 } else { 0 }));
    println!("{:?}", preds);
    Ok(())
}

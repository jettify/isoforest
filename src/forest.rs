use super::tree::{IsolationTree, IsolationTreeParams, MaxFeatures, MaxSamples};

use linfa::{
    dataset::DatasetBase,
    error::{Error, Result},
    traits::{Fit, Predict},
    Float,
};
use ndarray::{Array1, ArrayBase, Data, Ix2};

#[derive(Debug, Clone)]
pub struct IsolationForestParams {
    num_estimators: usize,
    tree_hyperparameters: IsolationTreeParams,
}

impl IsolationForestParams {
    pub fn new(num_estimators: usize, tree_params: &IsolationTreeParams) -> Self {
        Self {
            num_estimators,
            tree_hyperparameters: tree_params.clone(),
        }
    }
}

impl Default for IsolationForestParams {
    fn default() -> Self {
        Self {
            num_estimators: 100,
            tree_hyperparameters: IsolationTreeParams::default(),
        }
    }
}

impl IsolationForestParams {
    pub fn num_estimators(&self) -> usize {
        self.num_estimators
    }

    pub fn tree_hyperparameters(&self) -> IsolationTreeParams {
        self.tree_hyperparameters.clone()
    }

    pub fn with_num_estimators(mut self, n: usize) -> Self {
        self.num_estimators = n;
        self
    }

    pub fn with_tree_hyperparameters(mut self, params: IsolationTreeParams) -> Self {
        self.tree_hyperparameters = params;
        self
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct IsolationForest<F> {
    pub trees: Vec<IsolationTree<F>>,
}

impl<'a, F: Float, D: Data<Elem = F>, T> Fit<'a, ArrayBase<D, Ix2>, T> for IsolationForestParams {
    type Object = Result<IsolationForest<F>>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        let mut trees = Vec::with_capacity(self.num_estimators);
        self.validate()?;
        for i in 0..self.num_estimators {
            let new_seed: u64 = self.tree_hyperparameters.seed() + i as u64;
            let params = self.tree_hyperparameters.clone().with_seed(new_seed);
            let tree = params.with_seed(new_seed).fit(&dataset).unwrap();
            trees.push(tree);
        }
        let forest = IsolationForest { trees };
        Ok(forest)
    }
}

impl<F: Float, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Result<Array1<F>>>
    for IsolationForest<F>
{
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Result<Array1<F>> {
        let mut result: Array1<F> = Array1::zeros(x.nrows());
        let num_trees = F::from_usize(self.trees.len()).unwrap();
        for i in 0..self.trees.len() {
            result = result + self.trees[i].predict(&x) / (num_trees * self.trees[i].average_path);
        }
        result.mapv_inplace(|v| F::from_f32(2.0).unwrap().powf(-v));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::dataset::DatasetBase;
    use ndarray::array;

    #[test]
    fn test_hyperparameters() {
        let params = IsolationForestParams::default();
        assert_eq!(params.num_estimators, 100);
        assert_eq!(params.tree_hyperparameters.seed(), 0);
        let new_params = IsolationForestParams::default().with_num_estimators(10);
        assert!(new_params.validate().is_ok())
    }

    #[test]
    fn other_small_toy_dataset() {
        let data = array![
            [-2.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [6.0, 3.0],
            [-4.0, 7.0]
        ];

        let data = array![
            [-2.0],
            [-1.0],
            [1.0],
            [1.0],
            [2.0],
            [-1.5],
            [0.5],
            [-0.5],
            [4.0],
            [-5.0],
        ];

        let seed: u64 = 2;
        let dataset = DatasetBase::new(data.clone(), ());
        let sample_size = dataset.records().nrows();
        let num_features = dataset.records().ncols();
        let tree_params = IsolationTreeParams::new(
            MaxSamples::Absolute(sample_size),
            MaxFeatures::Absolute(num_features),
            seed,
        );

        let model = IsolationForestParams::new(1, &tree_params)
            .fit(&dataset)
            .unwrap();
        let data = array![[0.1],];

        let preds = model.predict(&data).unwrap();
        println!("{}", preds);
        //assert_eq!(preds.mapv(|a| if a > 0.5 { 1 } else { 0 }).sum(), 2);
    }

    #[test]
    fn toy_dataset() {
        let data = array![
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, -14.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 5.0, 3.0, 0.0, -4.0, 0.0, 0.0, 1.0, -5.0, 0.2, 0.0, 4.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, -4.5, 0.0, 0.0, 2.1, 1.0, 0.0, 0.0, -4.5, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
            [-1.0, -2.0, 0.0, 4.0, -3.0, 10.0, 4.0, 0.0, -3.2, 0.0, 4.0, 3.0, -4.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, 0.0,],
            [2.0, 8.0, 5.0, 1.0, 0.5, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 2.0, 0.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -2.0, 3.0, 0.0, 1.0, 0.0,],
            [2.0, 0.0, 1.0, 2.0, 3.0, -1.0, 10.0, 2.0, 0.0, -1.0, 1.0, 2.0, 2.0, 0.0,],
            [1.0, 1.0, 0.0, 2.0, 2.0, -1.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 3.0, 0.0,],
            [3.0, 1.0, 0.0, 3.0, 0.0, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 1.5, 1.0, -1.0, -1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 10.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, -1.0,],
            [2.0, 0.0, 5.0, 1.0, 0.5, -2.0, 10.0, 0.0, 1.0, -5.0, 3.0, 1.0, 0.0, -1.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -2.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0,],
            [2.0, 1.0, 1.0, 1.0, 2.0, -1.0, 10.0, 2.0, 0.0, -1.0, 0.0, 2.0, 1.0, 1.0,],
            [1.0, 1.0, 0.0, 0.0, 1.0, -3.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 1.0, 1.0,],
            [3.0, 1.0, 0.0, 1.0, 0.0, -4.0, 1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0, 0.0,]
        ];

        let targets = ();
        let dataset = DatasetBase::new(data.clone(), targets);
        let sample_size = dataset.records().nrows();
        let num_features = dataset.records().ncols();
        let seed = 1;
        let tree_params =
            IsolationTreeParams::new(MaxSamples::Ratio(0.5), MaxFeatures::Ratio(0.5), seed);

        let model = IsolationForestParams::new(100, &tree_params)
            .fit(&dataset)
            .unwrap();

        let preds = model.predict(&data).unwrap();
    }
}

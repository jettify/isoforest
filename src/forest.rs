use super::tree::{IsolationTree, IsolationTreeParams};

use linfa::{
    dataset::{DatasetBase, Targets},
    error::{Error, Result},
    traits::{Fit, Predict},
    Float,
};
use ndarray::{Array1, ArrayBase, Data, Ix2};
use ndarray_rand::rand::Rng;


#[derive(Debug, Clone)]
pub struct IsolationForestParams<R: Rng> {
    num_estimators: usize,
    tree_hyperparameters: IsolationTreeParams<R>,
}

impl<'a, R: Rng + Clone> IsolationForestParams<R> {
    pub fn new(num_estimators: usize, tree_params: &IsolationTreeParams<R>) -> Self {
        Self {
            num_estimators,
            tree_hyperparameters: tree_params.clone(),
        }
    }
}

impl<'a, R: Rng + Clone> IsolationForestParams<R> {
    pub fn num_estimators(&self) -> usize {
        self.num_estimators
    }

    pub fn tree_hyperparameters(&self) -> IsolationTreeParams<R> {
        self.tree_hyperparameters.clone()
    }

    pub fn with_num_estimators(mut self, n: usize) -> Self {
        self.num_estimators = n;
        self
    }

    pub fn with_tree_hyperparameters(mut self, params: IsolationTreeParams<R>) -> Self {
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

impl<'a, F: Float, R: Rng + Clone, D: Data<Elem = F>, T: Targets> Fit<'a, ArrayBase<D, Ix2>, T>
    for IsolationForestParams<R>
{
    type Object = Result<IsolationForest<F>>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        let mut trees = Vec::with_capacity(self.num_estimators);
        self.validate()?;
        for _ in 0..self.num_estimators {
            let tree = self.tree_hyperparameters
                .fit(&dataset).unwrap();
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
            result = result + self.trees[i].predict(&x) / (num_trees*self.trees[i].average_path);
        }
        result.mapv_inplace(|v| {F::from_f32(2.0).unwrap().powf(-v)});
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::dataset::DatasetBase;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

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

        let rng = Isaac64Rng::seed_from_u64(4);
        let dataset = DatasetBase::new(data.clone(), ());
        let sample_size = dataset.records().nrows();
        let num_features = dataset.records().ncols();
        let tree_params = IsolationTreeParams::new(sample_size, num_features, rng);

        let model = IsolationForestParams::new(5, &tree_params)
            .fit(&dataset)
            .unwrap();

        let preds = model.predict(&data).unwrap();
        assert_eq!(preds.mapv(|a| if a > 0.5 {1} else {0}).sum(), 2);
    }
}

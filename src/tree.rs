use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_rand::{rand::seq::SliceRandom, rand::Rng, rand_distr::Uniform};
use std::cell::RefCell;
use std::fmt::Debug;

use linfa::{
    dataset::DatasetBase,
    error::{Error, Result},
    traits::{Fit, Predict},
    Float,
};

const EULER_GAMMA: f64 = 0.57721566490153286060;

#[derive(Clone, Debug)]
pub struct IsolationTreeParams<R: Rng> {
    max_samples: usize,
    max_features: usize,
    rng: RefCell<R>,
}

impl<'a, R: Rng + Clone> IsolationTreeParams<R> {
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> IsolationTreeParams<R2> {
        IsolationTreeParams {
            max_samples: self.max_samples,
            max_features: self.max_features,
            rng: RefCell::new(rng),
        }
    }

    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    pub fn max_features(&self) -> usize {
        self.max_features
    }

    pub fn max_depth(&self) -> usize {
        ((self.max_samples as f64).max(2.0)).log2().ceil() as usize
    }

    pub fn cloned_rng(&mut self) -> R {
        self.rng.borrow().clone()
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl<'a, R: Rng + Clone> IsolationTreeParams<R> {
    pub fn new(max_samples: usize, max_features: usize, rng: R) -> Self {
        Self {
            max_samples,
            max_features,
            rng: RefCell::new(rng),
        }
    }
}

#[derive(Clone, Debug)]
struct ColumnsSubset {
    columns: Vec<usize>,
}

impl ColumnsSubset {
    fn new_random_subset<R: Rng>(num_columns: usize, select_num: usize, rng: &mut R) -> Self {
        let features: Vec<usize> = (0..num_columns).collect();
        let columns: Vec<usize> = features.choose_multiple(rng, select_num).cloned().collect();
        Self::new(columns)
    }

    fn new(columns: Vec<usize>) -> Self {
        ColumnsSubset { columns }
    }

    fn random_column<R: Rng>(&self, rng: &mut R) -> usize {
        *self.columns.choose(rng).unwrap()
    }
}

struct RowMask {
    mask: Vec<bool>,
    nsamples: usize,
}

impl RowMask {
    fn all(nsamples: usize) -> Self {
        RowMask {
            mask: vec![true; nsamples as usize],
            nsamples,
        }
    }

    fn none(nsamples: usize) -> Self {
        RowMask {
            mask: vec![false; nsamples as usize],
            nsamples: 0,
        }
    }

    fn mark(&mut self, idx: usize) {
        self.mask[idx] = true;
        self.nsamples += 1;
    }

    fn random_sample<R: Rng>(&mut self, sample_size: usize, rng: &mut R) {
        for i in 0..self.mask.len() {
            if i <= sample_size {
                self.mask[i] = true
            } else {
                self.mask[i] = false
            }
        }
        if sample_size < self.nsamples {
            self.mask.shuffle(rng)
        }
    }
}

struct SortedIndex<F: Float> {
    sorted_values: Vec<(usize, F)>,
}

impl<F: Float> SortedIndex<F> {
    fn of_array_column(x: &ArrayBase<impl Data<Elem = F>, Ix2>, feature_idx: usize) -> Self {
        let sliced_column: Vec<F> = x.index_axis(Axis(1), feature_idx).to_vec();
        let mut pairs: Vec<(usize, F)> = sliced_column.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

        SortedIndex {
            sorted_values: pairs,
        }
    }
    fn min(&self, mask: &RowMask) -> F {
        let min_idx = self
            .sorted_values
            .iter()
            .position(|&e| mask.mask[e.0])
            .unwrap();
        self.sorted_values[min_idx].1
    }

    fn max(&self, mask: &RowMask) -> F {
        let min_idx = self
            .sorted_values
            .iter()
            .rposition(|&e| mask.mask[e.0])
            .unwrap();
        self.sorted_values[min_idx].1
    }
}

#[derive(Debug, Clone)]
pub struct TreeNode<F> {
    feature_idx: usize,
    split_value: F,
    left_child: Option<Box<TreeNode<F>>>,
    right_child: Option<Box<TreeNode<F>>>,
    leaf_node: bool,
    depth: usize,
    size: usize,
}

impl<'a, F: Float + std::fmt::Debug> TreeNode<F> {
    pub fn params<R: Rng + Clone>(
        max_samples: usize,
        max_features: usize,
        rng: R,
    ) -> IsolationTreeParams<R> {
        IsolationTreeParams::new(max_samples, max_features, rng)
    }

    fn empty_leaf(depth: usize, size: usize) -> Self {
        TreeNode {
            feature_idx: 0,
            split_value: F::zero(),
            left_child: None,
            right_child: None,
            leaf_node: true,
            depth,
            size,
        }
    }

    fn fit<D: Data<Elem = F>, T, R: Rng + Clone>(
        data: &DatasetBase<ArrayBase<D, Ix2>, T>,
        mask: RowMask,
        hyperparameters: &IsolationTreeParams<R>,
        sorted_indices: &[SortedIndex<F>],
        columns: &ColumnsSubset,
        depth: usize,
        rng: &mut R,
    ) -> Self {
        if hyperparameters.max_depth() <= depth {
            return Self::empty_leaf(depth + 1, mask.nsamples);
        };
        if mask.nsamples < 2 {
            return Self::empty_leaf(depth + 1, mask.nsamples);
        }

        let col_idx = columns.random_column(rng);
        let feature_index = &sorted_indices[col_idx];

        let min = feature_index.min(&mask).to_f64().unwrap();
        let max = feature_index.max(&mask).to_f64().unwrap();
        if min == max {
            return Self::empty_leaf(depth + 1, mask.nsamples);
        }
        let split_value = rng.gen_range(min, max);

        let observations = data.records().view().nrows();

        let mut left_mask = RowMask::none(observations);
        let mut right_mask = RowMask::none(observations);

        for i in 0..observations {
            if mask.mask[i] {
                if data.records()[(i, col_idx)] <= F::from(split_value).unwrap() {
                    left_mask.mark(i);
                } else {
                    right_mask.mark(i);
                }
            }
        }

        let left_child = Some(Box::new(TreeNode::fit(
            data,
            left_mask,
            &hyperparameters,
            &sorted_indices,
            &columns,
            depth + 1,
            rng,
        )));

        let right_child = Some(Box::new(TreeNode::fit(
            data,
            right_mask,
            &hyperparameters,
            &sorted_indices,
            &columns,
            depth + 1,
            rng,
        )));

        TreeNode {
            feature_idx: col_idx,
            split_value: F::from(split_value).unwrap(),
            left_child,
            right_child,
            leaf_node: false,
            depth,
            size: mask.nsamples,
        }
    }
}

fn average_path_length<F: Float>(nsamples: usize) -> F {
    let n = nsamples as f64;
    let v: f64 = if nsamples > 2 {
        2.0 * (f64::ln(n - 1.0) + EULER_GAMMA) - (2.0 * (n - 1.0) / n)
    } else if nsamples == 2 {
        1.0
    } else {
        0.0
    };
    F::from_f64(v).unwrap()
}

fn find_path_length<'a, F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    node: &'a TreeNode<F>,
    d: usize,
) -> F {
    if node.leaf_node {
        F::from_usize(d).unwrap() + average_path_length::<F>(node.size)
    } else if x[node.feature_idx] < node.split_value {
        find_path_length(x, node.left_child.as_ref().unwrap(), d + 1)
    } else {
        find_path_length(x, node.right_child.as_ref().unwrap(), d + 1)
    }
}

#[derive(Clone, Debug)]
pub struct IsolationTree<F> {
    pub root_node: TreeNode<F>,
    pub average_path: F,
}

impl<'a, F: Float, R: Rng + Clone, D: Data<Elem = F>, T> Fit<'a, ArrayBase<D, Ix2>, T>
    for IsolationTreeParams<R>
{
    type Object = Result<IsolationTree<F>>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        self.validate()?;

        let x = dataset.records();
        let ref mut rng = *self.rng.borrow_mut();

        let mut training_sample = RowMask::all(x.nrows());
        training_sample.random_sample(self.max_samples(), rng);

        let sorted_indices: Vec<_> = (0..(x.ncols()))
            .map(|feature_idx| SortedIndex::of_array_column(&x, feature_idx))
            .collect();

        let columns = ColumnsSubset::new_random_subset(x.ncols(), self.max_features(), rng);

        let root_node = TreeNode::fit(
            &dataset,
            training_sample,
            &self,
            &sorted_indices,
            &columns,
            0,
            rng,
        );
        Ok(IsolationTree {
            root_node,
            average_path: average_path_length(self.max_samples()),
        })
    }
}

impl<F: Float, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Array1<F>> for IsolationTree<F> {
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        let result: Vec<F> = x
            .genrows()
            .into_iter()
            .map(|row| find_path_length(&row, &self.root_node, 0))
            .collect();
        Array1::from(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use linfa::Float;
    use ndarray::{array, Array};
    use ndarray_rand::{rand::SeedableRng, RandomExt};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn basic_tree_hyperparameters() {
        use ndarray_rand::rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        let rng = Isaac64Rng::seed_from_u64(42);
        let other_rng = Isaac64Rng::seed_from_u64(42);

        let params = IsolationTreeParams::new(1, 1, rng);
        let params = params
            .with_rng(other_rng)
            .with_max_samples(100)
            .with_max_features(5);

        let result = params.validate();
        assert!(result.is_ok());

        assert_eq!(params.max_samples(), 100);
        assert_eq!(params.max_features(), 5);
        assert_eq!(params.max_depth(), 7);
    }

    #[test]
    fn test_average_path_lenght() {
        assert_relative_eq!(average_path_length::<f64>(0), 0.0);
        assert_relative_eq!(average_path_length::<f64>(1), 0.0);
        assert_relative_eq!(average_path_length::<f64>(2), 1.0);
        assert_relative_eq!(average_path_length::<f64>(5), 2.327020052042847);
        assert_relative_eq!(average_path_length::<f64>(998), 12.965936877742774);
    }

    #[test]
    fn basic_tree() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let data = Array::random_using((50, 10), Uniform::new(-1., 1.), &mut rng);
        let dataset = DatasetBase::new(data, ());
        let params = IsolationTreeParams::new(50, 10, rng.clone());
        let tree = params.fit(&dataset);
        assert!(tree.is_ok());
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
        let rng = Isaac64Rng::seed_from_u64(42);
        let dataset = DatasetBase::new(data, targets);
        let sample_size = dataset.records().nrows();
        let num_features = dataset.records().ncols();
        let tree = IsolationTreeParams::new(sample_size, num_features, rng).fit(&dataset);
        assert!(tree.is_ok());
    }
}

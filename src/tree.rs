use ndarray::{s, Array1, ArrayBase, ArrayView1, ArrayViewMut1, Axis, Data, Ix1, Ix2, Slice};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand::seq::SliceRandom, rand::Rng, RandomExt};
use rand_isaac::Isaac64Rng;
use std::fmt::Debug;

use linfa::{
    dataset::DatasetBase,
    error::Result,
    traits::{Fit, Predict},
    Float,
};

const EULER_GAMMA: f64 = 0.57721566490153286060;

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

#[derive(Copy, Clone, Debug)]
pub enum MaxFeatures {
    Ratio(f32),
    Absolute(usize),
}

#[derive(Copy, Clone, Debug)]
pub enum MaxSamples {
    Auto,
    Absolute(usize),
    Ratio(f32),
}

#[derive(Clone, Debug)]
pub struct IsolationTreeParams {
    max_samples: MaxSamples,
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for IsolationTreeParams {
    fn default() -> Self {
        Self {
            max_samples: MaxSamples::Auto,
            max_features: MaxFeatures::Ratio(1.0),
            seed: 0,
        }
    }
}

impl IsolationTreeParams {
    pub fn new(max_samples: MaxSamples, max_features: MaxFeatures, seed: u64) -> Self {
        Self {
            max_samples,
            max_features,
            seed,
        }
    }
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> IsolationTreeParams {
        self.seed = seed;
        self
    }

    pub fn max_samples(&self) -> MaxSamples {
        self.max_samples
    }

    pub fn max_features(&self) -> MaxFeatures {
        self.max_features
    }

    pub fn max_depth(&self, max_samples: usize) -> usize {
        ((max_samples as f64).max(2.0)).log2().ceil() as usize
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn num_features(&self, ncols: usize) -> usize {
        match self.max_features {
            MaxFeatures::Ratio(ratio) => clip((ncols as f32 * ratio) as usize, 1, ncols),
            MaxFeatures::Absolute(num) => clip(num, 1, ncols),
        }
    }

    pub fn num_samples(&self, nrows: usize) -> usize {
        match self.max_samples {
            MaxSamples::Auto => usize::min(256, nrows),
            MaxSamples::Ratio(ratio) => clip((nrows as f32 * ratio).ceil() as usize, 1, nrows),
            MaxSamples::Absolute(num) => clip(num, 1, nrows),
        }
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct TreeSpace {
    start: usize,
    end: usize,
    depth: usize,
    is_left: bool,
    parent: Option<usize>,
}

impl TreeSpace {
    fn new(start: usize, end: usize, depth: usize, is_left: bool, parent: Option<usize>) -> Self {
        TreeSpace {
            start,
            end,
            depth,
            is_left,
            parent,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IsolationTree<F> {
    pub nodes: Vec<usize>,
    pub left_child: Vec<Option<usize>>,
    pub right_child: Vec<Option<usize>>,
    pub feature: Vec<usize>,
    pub split_value: Vec<F>,
    pub depth: Vec<usize>,
    pub size: Vec<usize>,
    pub average_path: F,
}

fn find_path_length<'a, F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    tree: &'a IsolationTree<F>,
    node_id: usize,
    d: usize,
) -> F {
    if tree.is_leaf(node_id) {
        F::from_usize(d).unwrap() + average_path_length::<F>(tree.size[node_id])
    } else if x[tree.feature[node_id]] < tree.split_value[node_id] {
        find_path_length(x, tree, tree.left_child[node_id].unwrap(), d + 1)
    } else {
        find_path_length(x, tree, tree.right_child[node_id].unwrap(), d + 1)
    }
}

impl<F: Float, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Array1<F>> for IsolationTree<F> {
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        let result: Vec<F> = x
            .genrows()
            .into_iter()
            .map(|row| find_path_length(&row, &self, 0, 0))
            .collect();
        Array1::from(result)
    }
}

impl<F: Float> IsolationTree<F> {
    fn is_leaf(&self, node_id: usize) -> bool {
        self.left_child[node_id].is_none() && self.right_child[node_id].is_none()
    }

    fn add_node(
        &mut self,
        feature: usize,
        split_value: F,
        depth: usize,
        size: usize,
        parent: Option<usize>,
        is_left: bool,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(idx);
        self.feature.push(feature);
        self.split_value.push(split_value);
        self.depth.push(depth);
        self.size.push(size);
        self.left_child.push(None);
        self.right_child.push(None);

        if let Some(parent_id) = parent {
            if is_left {
                self.left_child[parent_id] = Some(idx);
            } else {
                self.right_child[parent_id] = Some(idx);
            }
        }
        idx
    }

    fn add_leaf(
        &mut self,
        depth: usize,
        size: usize,
        parent: Option<usize>,
        is_left: bool,
    ) -> usize {
        self.add_node(0, F::zero(), depth, size, parent, is_left)
    }

    fn default() -> Self {
        let v: Vec<F> = Vec::new();
        IsolationTree {
            nodes: vec![],
            left_child: vec![],
            right_child: vec![],
            feature: vec![],
            split_value: v,
            depth: vec![],
            size: vec![],
            average_path: F::zero(),
        }
    }
}

fn clip(v: usize, min: usize, max: usize) -> usize {
    usize::min(usize::max(min, v), max)
}

impl<'a, F: Float, D: Data<Elem = F>, T> Fit<'a, ArrayBase<D, Ix2>, T> for IsolationTreeParams {
    type Object = Result<IsolationTree<F>>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        self.validate()?;
        let x = dataset.records();
        let nrows = dataset.records().nrows();
        let cols = dataset.records().ncols();

        let mut rng = Isaac64Rng::seed_from_u64(self.seed());
        let max_features = self.num_features(cols);
        let max_samples = self.num_samples(nrows);

        let mut sample_vec = (0..nrows).collect::<Vec<_>>();
        sample_vec.shuffle(&mut rng);
        sample_vec.truncate(max_samples);
        let sample_vec = Array1::from(sample_vec);
        let mut sample: Array1<usize> = Array1::from(sample_vec);

        let mut features_vec = (0..cols).collect::<Vec<_>>();
        features_vec.shuffle(&mut rng);
        features_vec.truncate(max_features);
        let features = Array1::from(features_vec);

        let mut rng = Isaac64Rng::seed_from_u64(self.seed());

        let mut stack: Vec<TreeSpace> = vec![TreeSpace::new(0, sample.len(), 0, true, None)];
        let mut tree = IsolationTree::default();
        tree.average_path = average_path_length(sample.len());

        while stack.len() != 0 {
            let rec: TreeSpace = stack.pop().unwrap();
            if rec.depth >= self.max_depth(max_samples) {
                tree.add_leaf(rec.depth, rec.end - rec.start, rec.parent, rec.is_left);
                continue;
            }
            if rec.end - rec.start <= 1 {
                tree.add_leaf(rec.depth, rec.end - rec.start, rec.parent, rec.is_left);
                continue;
            }

            let space = sample.slice_axis_mut(Axis(0), Slice::from(rec.start..rec.end));

            let feature: usize = features[rng.gen_range(0, features.len())];
            let min_idx = space
                .iter()
                .min_by(|l, r| x[[**l, feature]].partial_cmp(&x[[**r, feature]]).unwrap())
                .unwrap();

            let max_idx = space
                .iter()
                .max_by(|l, r| x[[**l, feature]].partial_cmp(&x[[**r, feature]]).unwrap())
                .unwrap();

            let min = x[[*min_idx, feature]].to_f64().unwrap();
            let max = x[[*max_idx, feature]].to_f64().unwrap();

            if &min == &max {
                tree.add_leaf(rec.depth, rec.end - rec.start, rec.parent, rec.is_left);
                continue;
            }

            let split_value = F::from_f64(rng.gen_range(min, max)).unwrap();
            let split_pos: usize = rec.start
                + update_sample(
                    &x.slice(s![.., feature]),
                    &mut sample.slice_mut(s![rec.start..rec.end]),
                    split_value,
                );

            let parent = tree.add_node(
                feature,
                split_value,
                rec.depth,
                rec.end - rec.start,
                rec.parent,
                rec.is_left,
            );

            stack.push(TreeSpace::new(
                rec.start,
                split_pos,
                rec.depth + 1,
                true,
                Some(parent),
            ));
            stack.push(TreeSpace::new(
                split_pos,
                rec.end,
                rec.depth + 1,
                false,
                Some(parent),
            ));
        }
        Ok(tree)
    }
}

fn update_sample<F: Float>(
    data: &ArrayView1<F>,
    sample: &mut ArrayViewMut1<usize>,
    threshold: F,
) -> usize {
    let mut start: usize = 0;
    let mut end: usize = sample.len();

    while start < end {
        if data[sample[start]] < threshold {
            start += 1;
        } else {
            end -= 1;
            sample.swap(start, end);
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_average_path_lenght() {
        assert_relative_eq!(average_path_length::<f64>(0), 0.0);
        assert_relative_eq!(average_path_length::<f64>(1), 0.0);
        assert_relative_eq!(average_path_length::<f64>(2), 1.0);
        assert_relative_eq!(average_path_length::<f64>(5), 2.327020052042847);
        assert_relative_eq!(average_path_length::<f64>(998), 12.965936877742774);
    }

    #[test]
    fn basic_tree_hyperparameters() {
        let params = IsolationTreeParams::default();
        let params = params
            .with_seed(42)
            .with_max_samples(MaxSamples::Absolute(100))
            .with_max_features(MaxFeatures::Absolute(10));

        let result = params.validate();
        assert!(result.is_ok());

        assert_eq!(params.max_depth(100), 7);
        assert_eq!(params.seed(), 42);
    }

    #[test]
    fn basic_data() {
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
        let seed: u64 = 2;
        let dataset = DatasetBase::new(data.clone(), ());
        let sample_size = dataset.records().nrows();
        let num_features = dataset.records().ncols();
        let params = IsolationTreeParams::new(
            MaxSamples::Absolute(sample_size),
            MaxFeatures::Absolute(num_features),
            seed,
        );
        let tree = params.fit(&dataset).unwrap();
        let preds = tree.predict(&data);
        assert_eq!(
            preds
                .mapv(|a| if a / tree.average_path > 0.0 { 1 } else { 0 })
                .sum(),
            8
        );
    }
}

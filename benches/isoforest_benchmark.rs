use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use isoforest::{IsolationForestParams, IsolationTreeParams, MaxFeatures, MaxSamples};
use linfa::prelude::*;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn isoforest_bench(c: &mut Criterion) {
    let training_set_sizes: Vec<usize> = vec![100, 1000, 10000, 100000];
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let tree_params =
        IsolationTreeParams::new(MaxSamples::Ratio(1.0), MaxFeatures::Ratio(1.0), seed);
    let hyperparams = IsolationForestParams::default().with_tree_hyperparameters(tree_params);

    let mut group = c.benchmark_group("isoforest");
    group.sample_size(10);
    let num_features = 25;

    for n in training_set_sizes.iter() {
        let train: Array2<f64> =
            Array2::random_using((*n, num_features), Uniform::new(0., 10.), &mut rng);

        let dataset = DatasetBase::new(train, ());
        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, d| {
            b.iter(|| hyperparams.fit(&d))
        });
    }

    group.finish();
}

criterion_group!(benches, isoforest_bench);
criterion_main!(benches);

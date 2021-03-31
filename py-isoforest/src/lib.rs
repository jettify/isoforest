use isoforest::{
    IsolationForest, IsolationForestParams, IsolationTreeParams, MaxFeatures, MaxSamples,
};
use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};
use ndarray::{array, Array1, ArrayView2};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rand_isaac::Isaac64Rng;
use std::io::Result;

#[pymodule]
fn _isoforest(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyclass(unsendable)]
    struct PyIsoForest {
        model: Option<IsolationForest<f64>>,
        params: IsolationForestParams,
    }

    #[pymethods]
    impl PyIsoForest {
        #[new]
        pub fn new<'py>(
            py: Python<'py>,
            num_estimators: usize,
            max_features: PyObject,
            sample_size: PyObject,
            seed: u64,
        ) -> Self {
            let num_features = if let Ok(num) = max_features.extract::<usize>(py) {
                MaxFeatures::Absolute(num)
            } else {
                let r: f32 = max_features.extract::<f32>(py).unwrap();
                MaxFeatures::Ratio(r)
            };

            let sample = if let Ok(num) = sample_size.extract::<usize>(py) {
                MaxSamples::Absolute(num)
            } else if let Ok(r) = sample_size.extract::<f32>(py) {
                MaxSamples::Ratio(r)
            } else {
                MaxSamples::Auto
            };

            let tree_params = IsolationTreeParams::new(sample, num_features, seed);
            let forest_params = IsolationForestParams::new(num_estimators, &tree_params);
            PyIsoForest {
                model: None,
                params: forest_params,
            }
        }

        pub fn predict<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<f64>,
        ) -> &'py PyArray1<f64> {
            let data = x.as_array();
            let m = self.model.as_ref().unwrap();
            let preds = m.predict(&data).unwrap();
            preds.into_pyarray(py)
        }

        pub fn decision_function<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<f64>,
        ) -> &'py PyArray1<f64> {
            let data = x.as_array();
            let m = self.model.as_ref().unwrap();
            let preds = m.decision_function(&data).unwrap();
            preds.into_pyarray(py)
        }


        pub fn fit<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) {
            let data = x.as_array();
            let dataset = DatasetBase::new(data, ());
            self.model = Some(self.params.fit(&dataset).unwrap());
        }
    }

    m.add_class::<PyIsoForest>()?;

    Ok(())
}

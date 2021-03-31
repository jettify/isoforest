import numpy as np
import pytest
# from sklearn.ensemble import IsolationForest as SklearnLIsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from isoforest import IsolationForest


def test_iforest_performance():
    """Test Isolation Forest performs well"""

    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)
    X_train = np.r_[X + 2, X - 2]
    X_train = X[:100]

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)

    # fit the model
    seed=3
    clf = IsolationForest(n_estimators=100, max_samples=100, seed=seed).fit(X_train)
    # predict scores (the lower, the more normal)
    y_pred =  -clf.decision_function(X_test)

    # check that there is at most 6 errors (false positive or false negative)
    assert roc_auc_score(y_test, y_pred) > 0.98


@pytest.mark.parametrize("contamination", [0.5])
def test_iforest_works(contamination):
    # toy sample (the last two samples are outliers)
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, 7]]).astype(float)

    # Test IsolationForest
    clf = IsolationForest(seed=1, contamination=contamination)
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])

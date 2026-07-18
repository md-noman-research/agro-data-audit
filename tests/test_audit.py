import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import data_audit  # noqa: F401


@pytest.fixture
def regression_df():
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 65, 200).astype(float),
        "Yield": np.random.normal(5, 1.5, 200),
        "Category": np.random.choice(["A", "B", "C"], 200),
    }
    data["Target_Reg"] = (
        data["Age"] * 2.5 + data["Yield"] * 10 + np.random.normal(0, 5, 200)
    )

    # Introduce missing values and outliers
    data["Age"][10] = np.nan
    data["Yield"][5] = 100.0  # Outlier
    df = pd.DataFrame(data)
    # Introduce duplicate row
    df = pd.concat([df, df.iloc[0:1]], ignore_index=True)
    return df


@pytest.fixture
def classification_df():
    np.random.seed(42)
    data = {
        "Score": np.random.normal(50, 15, 200),
        "Class": np.random.choice(["Pass", "Fail"], 200),
    }
    df = pd.DataFrame(data)
    return df


def test_scan_iqr(regression_df):
    audit = regression_df.audit
    issues = audit.scan(outlier_method="iqr")
    assert isinstance(issues, pd.DataFrame)
    assert not issues.empty
    problems = issues["Problem"].values
    assert "Duplicates" in problems
    assert "Missing" in problems
    assert "Outlier" in problems


def test_scan_zscore(regression_df):
    audit = regression_df.audit
    issues = audit.scan(outlier_method="zscore", zscore_thresh=2.0)
    assert isinstance(issues, pd.DataFrame)


def test_scan_custom_bounds(regression_df):
    audit = regression_df.audit
    issues = audit.scan(custom_bounds={"Yield": (0, 10)})
    assert isinstance(issues, pd.DataFrame)
    outlier_issues = issues[issues["Problem"] == "Outlier"]
    assert len(outlier_issues) > 0


def test_fix_suggest(regression_df):
    audit = regression_df.audit
    audit.scan()
    res = audit.fix(mode="suggest")
    assert "[SUGGEST]" in res
    assert regression_df["Age"].isnull().sum() > 0  # should not modify


def test_fix_auto(regression_df):
    audit = regression_df.audit
    audit.scan()
    res = audit.fix(mode="auto")
    assert "[APPLIED]" in res
    assert regression_df["Age"].isnull().sum() == 0
    assert regression_df.duplicated().sum() == 0


def test_fix_manual(regression_df):
    audit = regression_df.audit
    audit.scan()

    # Missing at index 10
    assert pd.isnull(regression_df.loc[10, "Age"])

    from data_audit import sid

    res = audit.fix(
        mode="manual",
        fixes=[sid((10, "Age"), 99.0), sid("Yield", 15.0), sid(1, "ignore")],
    )

    assert "[APPLIED]" in res
    assert "[IGNORED]" in res
    assert regression_df.loc[10, "Age"] == 99.0


def test_fix_before_scan(regression_df):
    audit = regression_df.audit
    res = audit.fix()
    assert "Run scan() first." in res


def test_summary(regression_df):
    audit = regression_df.audit
    res = audit.summary()
    assert "Rows: 201" in res
    assert "Cols: 4" in res


def test_report(regression_df):
    audit = regression_df.audit
    res = audit.report()
    assert "Run scan() first." in res
    audit.scan()
    res2 = audit.report()
    assert "Audit Report:" in res2


def test_anomaly(regression_df):
    audit = regression_df.audit
    # Using IsolationForest
    anomalies = audit.anomaly()
    assert isinstance(anomalies, pd.DataFrame)
    # Ensure it returns the correct row references cleanly


def test_history(regression_df):
    audit = regression_df.audit
    audit.scan()
    hist = audit.history()
    assert isinstance(hist, str)
    assert "Audit initialized." in hist


# MLModule Tests
def test_ml_train_evaluate_explain_regression(regression_df):
    audit = regression_df.audit
    # Train
    train_res = audit.ml.train(target="Target_Reg")
    assert "ML Trained: Regression" in train_res
    assert audit.ml.model is not None

    # Evaluate
    eval_res = audit.ml.evaluate()
    assert "RMSE" in eval_res
    assert "R2" in eval_res

    # Explain Global
    explain_res = audit.ml.explain()
    assert "Top" in explain_res

    # Explain Local (SHAP)
    local_data = {"Age": 30, "Yield": 5.0, "Category": "A"}
    local_explain = audit.ml.explain(local_data)
    assert (
        "Local Prediction Explanation" in local_explain
        or "SHAP unavailable" in local_explain
    )
    assert (
        "Prediction Drivers:" in local_explain
        or "Fallback to global importance." in local_explain
    )


def test_ml_train_evaluate_classification(classification_df):
    audit = classification_df.audit
    # Train
    train_res = audit.ml.train(target="Class")
    assert "ML Trained: Classification" in train_res

    # Evaluate
    eval_res = audit.ml.evaluate()
    assert "Accuracy" in eval_res
    assert "F1_Score" in eval_res
    assert "Confusion Matrix" in eval_res

    # Explain Local Classification
    local_data = {"Score": 50}
    local_explain = audit.ml.explain(local_data)
    assert (
        "Prediction Drivers:" in local_explain
        or "Fallback to global importance." in local_explain
    )


def test_ml_evaluate_before_train(regression_df):
    audit = regression_df.audit
    res = audit.ml.evaluate()
    assert "Train model first." in res


def test_ml_explain_before_train(regression_df):
    audit = regression_df.audit
    res = audit.ml.explain()
    assert "Train model first." in res


def test_ml_predict_missing_method(regression_df):
    audit = regression_df.audit
    audit.ml.train(target="Target_Reg")
    assert True


def test_ml_recommend(regression_df):
    audit = regression_df.audit
    res = audit.ml.recommend(target="Target_Reg")
    assert "Recommended Model: RandomForestRegressor" in res

    # Test invalid target
    res_invalid = audit.ml.recommend(target="Unknown")
    assert "not found in DataFrame" in res_invalid


# Boundary / Edge Case Tests
def test_empty_dataframe():
    df = pd.DataFrame()
    res = df.audit.scan()
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_all_nans_dataframe():
    df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    audit = df.audit
    res = audit.scan()
    assert len(res) == 3
    assert res["Problem"].iloc[1] == "Missing"

    # Try fixing all nans, should not crash, probably no-op or fills with NaN
    fix_res = audit.fix(mode="auto")
    assert "[APPLIED]" in fix_res or "[SUGGEST]" not in fix_res


def test_extreme_outlier_bounds():
    df = pd.DataFrame({"A": [1, 2, 3, 1000000000]})
    issues = df.audit.scan(outlier_method="zscore", zscore_thresh=1.0)
    assert issues["Problem"].iloc[0] == "Outlier"


# Profiling Tests
import sys
from unittest.mock import patch, MagicMock

def test_profile_engine_ydata_missing_dependency(regression_df):
    # Hide the ydata_profiling module if it exists
    with patch.dict(sys.modules, {'ydata_profiling': None}):
        with pytest.raises(ImportError, match="ydata-profiling is not installed"):
            regression_df.audit.profile()

def test_profile_engine_native_not_implemented(regression_df):
    with pytest.raises(NotImplementedError, match="Native lightweight profiling engine is planned"):
        regression_df.audit.profile(engine="native")

def test_profile_unknown_engine(regression_df):
    with pytest.raises(ValueError, match="Unknown profiling engine: unknown"):
        regression_df.audit.profile(engine="unknown")

@patch("builtins.print")
def test_profile_sampling(mock_print):
    df = pd.DataFrame({"A": range(10)})
    
    # Mock ydata_profiling so it doesn't try to import it
    mock_profile_class = MagicMock()
    mock_profile_instance = mock_profile_class.return_value
    
    mock_module = MagicMock()
    mock_module.ProfileReport = mock_profile_class
    
    with patch.dict(sys.modules, {'ydata_profiling': mock_module}):
        df.audit.profile(sample=5, output="dummy.html", title="Test", minimal=True)
        
        mock_print.assert_any_call("Dataset too large. Sampled 5 rows for profiling.")
        
        args, kwargs = mock_profile_class.call_args
        assert len(args[0]) == 5
        assert kwargs["title"] == "Test"
        assert kwargs["minimal"] is True
        
        mock_profile_instance.to_file.assert_called_once_with("dummy.html")

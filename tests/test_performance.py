import pytest
import pandas as pd
import numpy as np
import data_audit # noqa: F401

@pytest.fixture
def large_dataset():
    """Generates a 100,000 row dataset with various numeric and categorical features."""
    np.random.seed(42)
    n = 100000
    df = pd.DataFrame({
        "Age": np.random.randint(18, 80, n),
        "Salary": np.random.normal(50000, 15000, n),
        "Score": np.random.uniform(0, 100, n),
        "Category": np.random.choice(["A", "B", "C", "D"], n)
    })
    # Introduce some missing values and outliers
    df.loc[100:150, "Age"] = np.nan
    df.loc[500:550, "Salary"] = 1000000  # Outliers
    return df

def test_benchmark_audit_scan(benchmark, large_dataset):
    """
    Benchmarks our custom data_audit scanner.
    It checks for duplicates, missing values, and outliers using IQR.
    """
    def run_audit():
        return large_dataset.audit.scan()
        
    result = benchmark(run_audit)
    assert isinstance(result, pd.DataFrame)

def test_benchmark_pandas_native(benchmark, large_dataset):
    """
    Benchmarks a comparable manual Pandas approach.
    Checks missing values, duplicates, and basic stats (as a proxy for outlier scanning).
    """
    def run_native_pandas():
        # 1. Missing values
        missing = large_dataset.isnull().sum()
        # 2. Duplicates
        dupes = large_dataset.duplicated().sum()
        # 3. Stats / Outlier prep proxy
        stats = large_dataset.describe()
        
        # Simple IQR outlier check manually written to be fair
        q1 = large_dataset["Salary"].quantile(0.25)
        q3 = large_dataset["Salary"].quantile(0.75)
        iqr = q3 - q1
        outliers = large_dataset[
            (large_dataset["Salary"] < q1 - 1.5 * iqr) | 
            (large_dataset["Salary"] > q3 + 1.5 * iqr)
        ]
        
        return missing, dupes, stats, outliers
        
    result = benchmark(run_native_pandas)
    assert len(result) == 4

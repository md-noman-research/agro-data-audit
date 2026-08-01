import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def base_df():
    """Create a synthetic agricultural dataset for metamorphic testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "Farm_ID": range(1, 101),
        "Crop_Yield": np.random.normal(50, 10, 100),
        "Temperature_C": np.random.normal(25, 5, 100),
        "Soil_pH": np.random.normal(6.5, 0.5, 100)
    })

def inject_issues(df):
    """Inject some known issues into the dataset."""
    df_issues = df.copy()
    # Inject 5 missing values in Crop_Yield
    df_issues.loc[0:4, "Crop_Yield"] = np.nan
    # Inject 3 outliers in Temperature_C
    df_issues.loc[5:7, "Temperature_C"] = [100.0, 150.0, -50.0]
    return df_issues

def test_mr1_shuffling_invariance(base_df):
    """
    MR1: Shuffling Invariance
    Shuffling the rows of the dataset should NOT change the total number of issues found.
    """
    df_issues = inject_issues(base_df)
    
    # 1. Base execution
    base_scan = df_issues.audit.scan()
    base_issue_count = len(base_scan)
    
    # 2. Metamorphic execution (Shuffled)
    df_shuffled = df_issues.sample(frac=1, random_state=99).reset_index(drop=True)
    shuffled_scan = df_shuffled.audit.scan()
    shuffled_issue_count = len(shuffled_scan)
    
    # Assert relation holds
    assert base_issue_count == shuffled_issue_count

def test_mr2_scale_invariance(base_df):
    """
    MR2: Scale Invariance
    Scaling numeric columns by a constant factor should NOT change the Z-score outliers detected.
    """
    df_issues = inject_issues(base_df)
    
    # 1. Base execution
    # Only look at outlier issues
    base_scan = df_issues.audit.scan(outlier_method="zscore")
    base_outliers = base_scan[base_scan["Problem"] == "Outlier (zscore)"]
    base_outlier_count = len(base_outliers)
    
    # 2. Metamorphic execution (Scaled by 10)
    df_scaled = df_issues.copy()
    df_scaled["Temperature_C"] = df_scaled["Temperature_C"] * 10
    
    scaled_scan = df_scaled.audit.scan(outlier_method="zscore")
    scaled_outliers = scaled_scan[scaled_scan["Problem"] == "Outlier (zscore)"]
    scaled_outlier_count = len(scaled_outliers)
    
    # Assert relation holds
    assert base_outlier_count == scaled_outlier_count

def test_mr3_linear_duplication(base_df):
    """
    MR3: Linear Duplication
    Duplicating the dataset should exactly double the number of missing value issues.
    """
    df_issues = inject_issues(base_df)
    
    # 1. Base execution
    base_scan = df_issues.audit.scan()
    base_missing = base_scan[base_scan["Problem"] == "Missing Value"]
    base_missing_count = len(base_missing)
    
    # 2. Metamorphic execution (Duplicated dataset)
    df_duplicated = pd.concat([df_issues, df_issues], ignore_index=True)
    dup_scan = df_duplicated.audit.scan()
    dup_missing = dup_scan[dup_scan["Problem"] == "Missing Value"]
    dup_missing_count = len(dup_missing)
    
    # Assert relation holds
    assert dup_missing_count == base_missing_count * 2

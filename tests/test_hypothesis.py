import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column
import data_audit  # noqa: F401

# This uses Property-Based Testing (Hypothesis) which generates random, extreme edge-case DataFrames
# It acts like fuzzing for Python code.

@settings(max_examples=100, deadline=None)
@given(
    data_frames(
        columns=[
            column("Numeric1", elements=st.floats(allow_nan=True, allow_infinity=True)),
            column("Numeric2", elements=st.floats(allow_nan=True, allow_infinity=True)),
            column("Category", elements=st.sampled_from(["A", "B", "C", None]))
        ]
    )
)
def test_fuzz_audit_scan_and_fix(df):
    """
    Fuzz test: Ensures that scanning and auto-fixing a dataset NEVER crashes,
    even if the dataset is completely empty, contains infinite values, all NaNs, etc.
    """
    # 1. Scan the fuzzed DataFrame
    issues = df.audit.scan()
    assert isinstance(issues, pd.DataFrame)
    
    # 2. Fix the fuzzed DataFrame (Auto Mode)
    fix_result = df.audit.fix(mode='auto')
    assert isinstance(fix_result, str)
    
    # 3. Generate a report to ensure it doesn't crash on weird data
    report = df.audit.report()
    assert isinstance(report, str)

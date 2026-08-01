import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .ml import MLModule
from .report import summary_report, audit_report


@pd.api.extensions.register_dataframe_accessor("audit")
class DataAuditAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._issues = pd.DataFrame()
        self._history_log = ["Audit initialized."]
        self.ml = MLModule(self._obj, self._history_log)
        self._scan_completed = False

    def _validate_bounds(self, bounds):
        valid_bounds = {}
        if isinstance(bounds, dict):
            for col, b in bounds.items():
                if (
                    col in self._obj.columns
                    and isinstance(b, (list, tuple))
                    and len(b) == 2
                    and b[0] < b[1]
                ):
                    valid_bounds[col] = (b[0], b[1])
        return valid_bounds

    def scan(
        self,
        outlier_method="iqr",
        zscore_thresh=3.0,
        custom_bounds=None,
        issue_filter="all",
    ):
        """
        Scans the DataFrame for missing values, duplicates, and outliers.
        
        Args:
            outlier_method (str): The statistical method for outlier detection 
                ('iqr' or 'zscore'). Default is 'iqr'.
            zscore_thresh (float): The threshold for Z-score anomalies (default: 3.0).
            custom_bounds (dict, optional): Custom min/max bounds per column. 
                e.g., {'pH': (5.0, 8.5)}
            issue_filter (str): Filter for issues to include (default: 'all').
                
        Returns:
            pd.DataFrame: A DataFrame containing all identified issues, 
            or an empty DataFrame if the data is perfectly clean.
        """
        valid_bounds = self._validate_bounds(custom_bounds)
        issues, i_id = [], 1

        if self._obj.duplicated().sum() > 0:
            issues.append(
                {
                    "Issue ID": i_id,
                    "Column": "All",
                    "Problem": "Duplicates",
                    "Value": self._obj.duplicated().sum(),
                    "Method": "N/A",
                }
            )
            i_id += 1

        for col in self._obj.columns:
            missing = self._obj[col].isnull().sum()
            if missing > 0:
                issues.append(
                    {
                        "Issue ID": i_id,
                        "Column": col,
                        "Problem": "Missing",
                        "Value": missing,
                        "Method": "N/A",
                    }
                )
                i_id += 1

            if pd.api.types.is_numeric_dtype(self._obj[col]):
                clean = self._obj[col].dropna()
                if not clean.empty:
                    method_used = outlier_method
                    if col in valid_bounds:
                        min_v, max_v = valid_bounds[col]
                        outliers = self._obj[
                            (self._obj[col] < min_v) | (self._obj[col] > max_v)
                        ]
                        method_used = f"custom ({min_v} to {max_v})"
                    elif outlier_method == "iqr":
                        q1, q3 = clean.quantile([0.25, 0.75])
                        outliers = self._obj[
                            (self._obj[col] < q1 - 1.5 * (q3 - q1))
                            | (self._obj[col] > q3 + 1.5 * (q3 - q1))
                        ]
                    elif outlier_method == "zscore":
                        m, s = clean.mean(), clean.std()
                        if s > 0:
                            outliers = self._obj[
                                (self._obj[col] < m - zscore_thresh * s)
                                | (self._obj[col] > m + zscore_thresh * s)
                            ]

                    if not outliers.empty:
                        issues.append(
                            {
                                "Issue ID": i_id,
                                "Column": col,
                                "Problem": "Outlier",
                                "Value": len(outliers),
                                "Method": method_used,
                            }
                        )
                        i_id += 1

        columns = ["Issue ID", "Column", "Problem", "Value", "Method"]
        self._issues = pd.DataFrame(issues, columns=columns)
        self._scan_completed = True
        
        if self._issues.empty:
            print("✔ Data clean.")
            
        return self._issues

    def fix(
        self,
        sid=None,
        outlier_method="iqr",
        zscore_thresh=3.0,
        custom_bounds=None,
        issue_filter="all",
        mode="auto",
        fixes=None,
    ):
        """
        Fixes identified data quality issues based on the specified mode.
        
        Args:
            sid (int, optional): Specific Issue ID to fix.
            outlier_method (str): Method to detect outliers ('iqr' or 'zscore').
            zscore_thresh (float): Threshold for z-score outlier detection.
            custom_bounds (dict, optional): Custom min/max bounds for specific columns.
            issue_filter (str): Filter for issues to include.
            mode (str): The fixing strategy to employ.
                - 'auto': Automatically fills missing values and clips outliers.
                - 'suggest': Returns suggestions for fixes without applying them.
                - 'manual': Applies targeted manual fixes using the `sid` helper.
            fixes (list of namedtuple, optional): Required if mode='manual'. 
                A list of `sid` objects detailing exactly what to replace.
                
        Returns:
            str: A formatted log of all fixes that were applied or suggested.
        """
        if not self._scan_completed:
            return "Run scan() first."
        valid_bounds = self._validate_bounds(custom_bounds)
        fix_list = (
            self._issues
            if sid is None
            else self._issues[self._issues["Issue ID"] == sid]
        )
        res = []

        if mode == "manual" and fixes and isinstance(fixes, list):
            for fix_item in fixes:
                k, v = fix_item.id, fix_item.value
                if str(v).lower() == "ignore":
                    res.append(f"[IGNORED] Manual fix for ID {k}")
                    continue

                if isinstance(k, tuple) and len(k) == 2:
                    row_idx, col_name = k
                    if col_name in self._obj.columns:
                        self._obj.loc[row_idx, col_name] = v
                        res.append(
                            f"[APPLIED] Manual fix for '{col_name}' at index {row_idx}"
                        )
                elif isinstance(k, int):
                    issue_row = self._issues[self._issues["Issue ID"] == k]
                    if not issue_row.empty:
                        col_name = issue_row["Column"].iloc[0]
                        prob = issue_row["Problem"].iloc[0]
                        if prob == "Missing":
                            self._obj[col_name] = self._obj[col_name].fillna(v)
                            res.append(
                                f"[APPLIED] Manual fill '{col_name}' (Issue {k})"
                            )
                elif isinstance(k, str) and k in self._obj.columns:
                    self._obj[k] = self._obj[k].fillna(v)
                    res.append(f"[APPLIED] Manual fill '{k}'")
            return "\n".join(res)

        if (fix_list["Problem"] == "Duplicates").any():
            if mode == "suggest":
                res.append("[SUGGEST] Drop duplicates")
            else:
                self._obj.drop_duplicates(inplace=True)
                self._obj.reset_index(drop=True, inplace=True)
                res.append("[APPLIED] Drop duplicates")

        for _, row in fix_list.iterrows():
            col, prob = row["Column"], row["Problem"]
            if prob == "Missing":
                val = (
                    self._obj[col].median()
                    if pd.api.types.is_numeric_dtype(self._obj[col])
                    else self._obj[col].mode()[0]
                )
                if mode == "suggest":
                    res.append(f"[SUGGEST] Fill '{col}' with {val}")
                else:
                    self._obj.fillna({col: val}, inplace=True)
                    res.append(f"[APPLIED] Fill '{col}'")
            elif prob == "Outlier":
                # Re-calculate clip bounds
                clean = self._obj[col].dropna()
                lower, upper = None, None
                if col in valid_bounds:
                    lower, upper = valid_bounds[col]
                elif outlier_method == "iqr":
                    q1, q3 = clean.quantile([0.25, 0.75])
                    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                elif outlier_method == "zscore":
                    m, s = clean.mean(), clean.std()
                    if s > 0:
                        lower, upper = m - zscore_thresh * s, m + zscore_thresh * s

                if lower is not None:
                    if mode == "suggest":
                        res.append(f"[SUGGEST] Clip '{col}'")
                    else:
                        self._obj.loc[:, col] = self._obj[col].clip(lower, upper)
                        res.append(f"[APPLIED] Clip '{col}'")

        return "\n".join(res)

    def summary(self):
        """
        Generates a quick statistical summary of the dataset.
        
        Returns:
            str: Basic dataset statistics including row/col counts and memory usage.
        """
        return summary_report(self._obj)

    def report(self, report_format="text"):
        """
        Generates a comprehensive audit and Machine Learning report.
        
        Args:
            report_format (str): The format of the report. Currently supports 'text'.
            
        Returns:
            str: A textual report of total issues, ML model metrics, and dataset shape.
        """
        if not self._scan_completed:
            return "Run scan() first."
        return audit_report(self._obj, self._issues, self.ml)

    def anomaly(self):
        """
        Detects severe multivariate anomalies using Isolation Forest.
        
        Returns:
            pd.DataFrame: A subset of the original DataFrame containing only 
            the highly unusual/anomalous rows (top 1%).
        """
        num = self._obj.select_dtypes(include=[np.number]).dropna()
        if num.empty:
            return pd.DataFrame()
        iso = IsolationForest(contamination=0.01, n_jobs=-1, random_state=42).fit(num)
        anomaly_indices = num.index[iso.predict(num) == -1]
        return self._obj.loc[anomaly_indices]

    def history(self):
        """
        Retrieves the audit trail log of operations performed on the dataset.
        
        Returns:
            str: A chronological log of actions like scanning, fixing, and ML training.
        """
        return "\n".join(self._history_log)

    def profile(
        self,
        output="audit_report.html",
        title="Data Audit Profile Report",
        minimal=False,
        sample=100000,
        engine="ydata"
    ):
        """
        Generates a comprehensive Exploratory Data Analysis (EDA) HTML report.
        
        Args:
            output (str): The filename for the HTML report (default: 'audit_report.html').
            title (str): The title displayed inside the report.
            minimal (bool): If True, skips expensive correlation calculations.
            sample (int): Automatically samples the DataFrame if it exceeds this 
                number of rows to prevent memory crashes. Set to None to disable.
            engine (str): The profiling engine to use (default: 'ydata').
            
        Returns:
            ProfileReport: The generated profiling object.
            
        Raises:
            ImportError: If the optional 'ydata-profiling' package is not installed.
        """
        # 1. Large dataset safety (Sampling)
        if sample and len(self._obj) > sample:
            df_to_profile = self._obj.sample(n=sample, random_state=42)
            print(f"Dataset too large. Sampled {sample} rows for profiling.")
        else:
            df_to_profile = self._obj

        # 2. Engine architecture
        if engine == "ydata":
            try:
                from ydata_profiling import ProfileReport
            except ImportError:
                raise ImportError(
                    "ydata-profiling is not installed. "
                    "Install it with: pip install agro-data-intelligence[profile]"
                )
            
            profile_obj = ProfileReport(df_to_profile, title=title, minimal=minimal)
            profile_obj.to_file(output)
            print(f"✔ Profile report saved to {output}")
            return profile_obj
        
        elif engine == "native":
            raise NotImplementedError("Native lightweight profiling engine is planned for a future release.")
        else:
            raise ValueError(f"Unknown profiling engine: {engine}")

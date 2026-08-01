def summary_report(df):
    """
    Generates a brief statistical summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        
    Returns:
        str: A formatted text string containing row/column counts, memory usage, 
        and transposed descriptive statistics.
    """
    mem = df.memory_usage(deep=False).sum() / 1024**2
    return (
        f"Rows: {len(df)} | Cols: {len(df.columns)} | Mem: {mem:.2f} MB\n"
        + df.describe().T.to_string()
    )


def audit_report(df, issues, ml_module, _report_format="text"):
    """
    Generates a comprehensive audit report for the dataset.
    
    Args:
        df (pd.DataFrame): The audited DataFrame.
        issues (pd.DataFrame): The DataFrame of identified data quality issues.
        ml_module (MLModule): The active MLModule instance to extract metrics from.
        report_format (str): The format of the report (currently supports 'text').
        
    Returns:
        str: A formatted text report summarizing data shape, issues found, 
        and the latest Machine Learning model metrics.
    """
    rep = f"Audit Report:\nRows: {len(df)}\nIssues: {len(issues)}\n"
    if ml_module.model is not None:
        rep += f"ML: {ml_module.task_type}\nMetrics: {ml_module.metrics}"
    return rep

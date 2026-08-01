import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


class MLModule:
    """
    ML Module: Preprocessing, Training, Evaluation, and Explainability.
    """

    def __init__(self, df, history_log):
        self._df = df
        self._history_log = history_log
        self.model = None
        self.target = None
        self.feature_list = []
        self.model_columns = None
        self.task_type = None

        self.num_cols = pd.Index([])
        self.cat_cols = pd.Index([])
        self.scaler = StandardScaler()
        self.label_encoder = None

        self.X_test = None
        self.y_test = None
        self.metrics = {}
        self.num_imputes = pd.Series(dtype=float)
        self.cat_imputes = pd.Series(dtype=object)

    @staticmethod
    def _get_task_type(target_series):
        if (
            pd.api.types.is_numeric_dtype(target_series)
            and target_series.nunique() >= 20
        ):
            return "Regression"
        return "Classification"

    def recommend(self, target):
        """
        Analyzes the target variable and recommends the best ML model.
        
        Args:
            target (str): The column name to predict.
            
        Returns:
            str: A formatted recommendation report detailing the task type 
            (Regression or Classification), the recommended model, and the 
            primary metrics to watch.
        """
        if target not in self._df.columns:
            return f"Target '{target}' not found in DataFrame."

        task = MLModule._get_task_type(self._df[target])
        res = [f"--- ML Recommendation for '{target}' ---"]
        res.append(f"Detected Task Type: {task}")
        if task == "Regression":
            res.append("Recommended Model: RandomForestRegressor")
            res.append("Metrics to watch: R2 Score, RMSE, MAE")
        else:
            res.append("Recommended Model: RandomForestClassifier")
            res.append("Metrics to watch: Accuracy, F1-Score, Confusion Matrix")

        return "\n".join(res)

    def train(self, target, features=None, max_cardinality=50):
        """
        Automatically preprocesses data and trains a Machine Learning model.
        
        Args:
            target (str): The column name to predict.
            features (list, optional): List of feature columns to use. If None, auto-selects.
            max_cardinality (int): Drops categorical features with more unique values 
                than this limit to prevent memory/performance issues. Default is 50.
                
        Returns:
            str: A training status log including warnings for dropped columns 
            and the final count of features used.
        """
        self.target = target
        self.task_type = self._get_task_type(self._df[target])
        log_msgs = []

        missing_target = self._df[target].isnull().sum()
        if missing_target > 0:
            missing_pct = (missing_target / len(self._df)) * 100
            log_msgs.append(
                f"[Warning] Target '{target}' missing {missing_pct:.1f}%. Rows dropped."
            )

        if features is None:
            self.feature_list = []
            for c in self._df.columns:
                if c == target or pd.api.types.is_datetime64_any_dtype(self._df[c]):
                    continue
                if (
                    not pd.api.types.is_numeric_dtype(self._df[c])
                    and self._df[c].nunique() > max_cardinality
                ):
                    log_msgs.append(
                        f"[Info] Dropped '{c}' (Cardinality > {max_cardinality})"
                    )
                    continue
                self.feature_list.append(c)
        else:
            self.feature_list = features

        model_data = self._df[self.feature_list + [target]].dropna(subset=[target])
        X = model_data[self.feature_list].copy()
        y = model_data[target].copy()

        if self.task_type == "Classification":
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index)

        self.num_cols = X.select_dtypes(include=[np.number]).columns
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns

        if not self.num_cols.empty:
            self.num_imputes = X[self.num_cols].median()
            X.loc[:, self.num_cols] = self.scaler.fit_transform(
                X[self.num_cols].fillna(self.num_imputes)
            )

        if not self.cat_cols.empty:
            self.cat_imputes = X[self.cat_cols].mode().iloc[0]
            X.loc[:, self.cat_cols] = X[self.cat_cols].fillna(self.cat_imputes)

        X = pd.get_dummies(X, drop_first=True)
        self.model_columns = X.columns

        if len(X) > 100000:
            X, _, y, _ = train_test_split(X, y, train_size=100000, random_state=42)
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = (
            RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            if self.task_type == "Regression"
            else RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        )

        self.model.fit(X_train, y_train)

        self._history_log.append(f"Trained {self.task_type} model.")
        return (
            "\n".join(log_msgs)
            + f"\nML Trained: {self.task_type} | Features: {len(self.model_columns)}"
        )

    def evaluate(self):
        """
        Evaluates the trained model on a holdout test set.
        
        Returns:
            str: A formatted report containing performance metrics.
            For Regression: R2, RMSE, MAE.
            For Classification: Accuracy, F1-Score, Confusion Matrix.
        """
        if self.model is None:
            return "Train model first."
        y_pred = self.model.predict(self.X_test)
        res = [f"--- Model Evaluation ({self.task_type}) ---"]

        if self.task_type == "Regression":
            self.metrics["R2"] = r2_score(self.y_test, y_pred)
            self.metrics["RMSE"] = np.sqrt(mean_squared_error(self.y_test, y_pred))
            self.metrics["MAE"] = mean_absolute_error(self.y_test, y_pred)
            for k, v in self.metrics.items():
                res.append(f"{k}: {v:.4f}")
        else:
            self.metrics["Accuracy"] = accuracy_score(self.y_test, y_pred)
            self.metrics["F1_Score"] = f1_score(
                self.y_test, y_pred, average="macro", zero_division=0
            )
            for k, v in self.metrics.items():
                res.append(f"{k}: {v*100:.2f}%")
            res.append(
                "\nConfusion Matrix:\n" + str(confusion_matrix(self.y_test, y_pred))
            )

        return "\n".join(res)

    def top_features(self, top_n=5):
        """
        Retrieves the top N most important features from the trained model.
        
        Args:
            top_n (int): Number of features to return (default: 5).
            
        Returns:
            str: A formatted list of the top features and their percentage importance.
        """
        if self.model is None:
            return "Train model first."
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        res = [f"--- Top {top_n} Important Features ---"]
        for i in indices:
            res.append(f"{self.model_columns[i]}: {importances[i]*100:.2f}%")
        return "\n".join(res)

    def _preprocess_inference(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        X_new = data.reindex(columns=self.feature_list).copy()

        if not self.num_cols.empty:
            X_new.loc[:, self.num_cols] = self.scaler.transform(
                X_new[self.num_cols].fillna(self.num_imputes)
            )
        if not self.cat_cols.empty:
            X_new.loc[:, self.cat_cols] = X_new[self.cat_cols].fillna(self.cat_imputes)

        X_new = pd.get_dummies(X_new)
        return X_new.reindex(columns=self.model_columns, fill_value=0)

    def explain(self, local_data=None):
        """
        Explains model predictions using SHAP (SHapley Additive exPlanations).
        
        Args:
            local_data (pd.DataFrame or dict, optional): A single row of data to explain.
                If None, returns global top feature importances instead.
                
        Returns:
            str: A formatted explanation showing the exact impact of each feature 
            on the specific prediction.
        """
        if self.model is None:
            return "Train model first."
        if local_data is None:
            return self.top_features(top_n=10)

        res = ["--- Local Prediction Explanation (SHAP) ---"]
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            prep_data = self._preprocess_inference(local_data)
            shap_values = explainer.shap_values(prep_data)

            # Robust Multiclass Handling
            if self.task_type == "Classification":
                pred_label = self.model.predict(prep_data)[0]
                # Match predicted label to model class index
                class_idx = np.where(self.model.classes_ == pred_label)[0][0]

                # Extract SHAP value based on identified class index
                if isinstance(shap_values, list):
                    sv = shap_values[class_idx][0]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                    sv = shap_values[0, :, class_idx]
                else:
                    sv = shap_values[0]
            else:
                sv = shap_values[0]

            res.append("Prediction Drivers:")
            for idx, col in enumerate(self.model_columns):
                impact = sv[idx]
                if abs(impact) > 0.01:
                    res.append(f"{col}: {'+' if impact > 0 else ''}{impact:.3f}")
        except Exception as e:
            res.append(f"[Info] SHAP unavailable ({e}). Fallback to global importance.")
            res.append(self.top_features(top_n=3))
        return "\n".join(res)

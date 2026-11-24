from __future__ import annotations
from core.preprocessor import AdaptivePreprocessor

import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize

from models.base import BaseModel


class ModelEvaluator:
    """
    Motor de evaluación cuantitativa para modelos supervisados (clasificación y regresión).
    Calcula métricas formales, matrices de confusión normalizadas por filas, curvas ROC
    multiclase ancladas en el origen, curvas Precision-Recall, validación cruzada estratificada
    y curvas de aprendizaje mediante clonación profunda (deepcopy) compatible con ABC.
    """

    @staticmethod
    def evaluate_classification(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> dict:
        """
        Calcula métricas de clasificación y coordenadas para matriz de confusión normalizada,
        curvas ROC y curvas Precision-Recall (binarias y multiclase One-vs-Rest).
        """
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()

        if y_proba is not None:
            y_proba_arr = np.asarray(y_proba, dtype=float)
            if y_proba_arr.ndim == 1:
                y_proba_arr = np.column_stack([1.0 - y_proba_arr, y_proba_arr])
            n_proba_cols = y_proba_arr.shape[1]
            if class_names is not None and len(class_names) == n_proba_cols:
                classes = np.arange(n_proba_cols)
                resolved_names = [str(c) for c in class_names]
            else:
                inferred = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
                if len(inferred) == n_proba_cols:
                    classes = inferred
                else:
                    classes = np.arange(n_proba_cols)
                resolved_names = (
                    [str(c) for c in class_names]
                    if class_names is not None and len(class_names) == len(classes)
                    else [str(c) for c in classes]
                )
        else:
            y_proba_arr = None
            if class_names is not None:
                classes = np.arange(len(class_names))
                if not np.isin(y_true_arr, classes).all():
                    classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
                    resolved_names = [str(c) for c in classes]
                else:
                    resolved_names = [str(c) for c in class_names]
            else:
                classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
                resolved_names = [str(c) for c in classes]

        n_classes = len(classes)

        cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(
                cm_raw.astype(float),
                row_sums,
                out=np.zeros_like(cm_raw, dtype=float),
                where=row_sums != 0,
            )

        results: dict = {
            "task_type": "classification",
            "classes": classes,
            "class_names": resolved_names,
            "metrics": {
                "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
                "precision_weighted": float(
                    precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
                ),
                "recall_weighted": float(
                    recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
                ),
                "f1_weighted": float(
                    f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
                ),
            },
            "confusion_matrix_raw": cm_raw,
            "confusion_matrix_normalized": cm_normalized,
            "has_probabilities": y_proba_arr is not None,
            "roc_curves": None,
            "pr_curves": None,
        }

        if y_proba_arr is None:
            return results

        if n_classes == 2:
            y_bin = np.column_stack(
                [
                    (y_true_arr == classes[0]).astype(int),
                    (y_true_arr == classes[1]).astype(int),
                ]
            )
        else:
            y_bin = label_binarize(y_true_arr, classes=classes)

        roc_data = {}
        pr_data = {}
        mean_fpr = np.linspace(0.0, 1.0, 200)
        tprs_interp = []
        ap_scores = []

        for idx, cls_label in enumerate(resolved_names):
            pos_count = int(y_bin[:, idx].sum())
            neg_count = int(len(y_bin) - pos_count)
            if pos_count == 0 or neg_count == 0:
                warnings.warn(
                    f"La clase '{cls_label}' no tiene muestras positivas y negativas simultáneamente "
                    "en evaluación; se omite de las curvas ROC/PR."
                )
                continue

            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba_arr[:, idx])
            if fpr[0] != 0.0 or tpr[0] != 0.0:
                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.insert(tpr, 0, 0.0)
            roc_auc = float(auc(fpr, tpr))
            roc_data[cls_label] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_interp.append(interp_tpr)

            prec, rec, _ = precision_recall_curve(y_bin[:, idx], y_proba_arr[:, idx])
            ap = float(average_precision_score(y_bin[:, idx], y_proba_arr[:, idx]))
            pr_data[cls_label] = {"precision": prec, "recall": rec, "ap": ap}
            ap_scores.append(ap)

        if tprs_interp:
            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_tpr[-1] = 1.0
            macro_auc = float(auc(mean_fpr, mean_tpr))
            roc_data["macro_average"] = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": macro_auc}
            results["metrics"]["roc_auc_macro"] = macro_auc
            results["metrics"]["pr_auc_macro"] = float(np.mean(ap_scores))
            results["roc_curves"] = roc_data
            results["pr_curves"] = pr_data
        else:
            results["has_probabilities"] = False

        return results

    @staticmethod
    def evaluate_regression(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        confidence_z: float = 1.96,
    ) -> dict:
        """
        Calcula métricas formales de regresión (R2, RMSE, MAE), residuales
        y semi-ancho de banda de confianza (±z * std_residual).
        """
        y_true_arr = np.asarray(y_true, dtype=float).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

        residuals = y_true_arr - y_pred_arr
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 else 0.0
        res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        return {
            "task_type": "regression",
            "metrics": {
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "residual_std": res_std,
            },
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "residuals": residuals,
            "confidence_band_half_width": float(confidence_z * res_std),
        }

    @staticmethod
    def _build_cv_splitter(
        y_series: pd.Series, task_type: str, k_folds: int, random_state: int
    ):
        """Construye un particionador K-Fold seguro adaptado al tamaño y balance del dataset."""
        n_samples = len(y_series)
        if n_samples < 2:
            raise ValueError("Se requieren al menos 2 muestras para validación cruzada.")

        if task_type == "classification":
            min_class_count = int(y_series.value_counts().min())
            if min_class_count < 2:
                eff_folds = min(k_folds, n_samples)
                warnings.warn(
                    f"Clase minoritaria con {min_class_count} muestra; usando KFold({eff_folds}) sin estratificar."
                )
                return KFold(n_splits=eff_folds, shuffle=True, random_state=random_state)
            eff_folds = min(k_folds, min_class_count)
            if eff_folds < k_folds:
                warnings.warn(
                    f"Ajustando k_folds de {k_folds} a {eff_folds} por tamaño de la clase minoritaria."
                )
            return StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=random_state)

        if task_type == "regression":
            eff_folds = min(k_folds, n_samples)
            return KFold(n_splits=eff_folds, shuffle=True, random_state=random_state)

        raise ValueError("task_type debe ser 'classification' o 'regression'.")

    @staticmethod
    def cross_validate(
        model_template: BaseModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        task_type: str = "classification",
        k_folds: int = 5,
        random_state: int = 42,
        scale: bool | None = None,
    ) -> dict:
        """
        Ejecuta validación cruzada K-Fold clonando mediante deepcopy cualquier modelo
        que herede de BaseModel.
        """
        y_series = pd.Series(np.asarray(y).ravel(), index=X.index)
        splitter = ModelEvaluator._build_cv_splitter(y_series, task_type, k_folds, random_state)

        fold_records = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_series), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]

            if scale is not None:
                prep = AdaptivePreprocessor()
                X_tr = prep.fit_transform_train(X_tr, scale=scale)
                X_val = prep.transform_test(X_val)

            model_clone = copy.deepcopy(model_template)
            model_clone.fit(X_tr, y_tr)
            preds = model_clone.predict(X_val)

            if task_type == "classification":
                fold_records.append(
                    {
                        "fold": fold_idx,
                        "accuracy": float(accuracy_score(y_val, preds)),
                        "f1_weighted": float(
                            f1_score(y_val, preds, average="weighted", zero_division=0)
                        ),
                    }
                )
            else:
                r2_val = float(r2_score(y_val, preds)) if len(y_val) >= 2 else 0.0
                fold_records.append(
                    {
                        "fold": fold_idx,
                        "r2": r2_val,
                        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
                        "mae": float(mean_absolute_error(y_val, preds)),
                    }
                )

        df_folds = pd.DataFrame(fold_records)
        summary = {
            col: {
                "mean": float(df_folds[col].mean()),
                "std": float(df_folds[col].std(ddof=1)) if len(df_folds) > 1 else 0.0,
            }
            for col in df_folds.columns
            if col != "fold"
        }

        return {"folds_df": df_folds, "summary": summary}

    @staticmethod
    def compute_learning_curve(
        model_template: BaseModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        task_type: str = "classification",
        train_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
        k_folds: int = 5,
        random_state: int = 42,
        scale: bool | None = None,
    ) -> dict:
        """
        Calcula curvas de aprendizaje (Error de entrenamiento vs. Error de validación)
        barajando los índices de entrenamiento por fold para mantener balance de clases.
        """
        y_series = pd.Series(np.asarray(y).ravel(), index=X.index)
        splitter = ModelEvaluator._build_cv_splitter(y_series, task_type, k_folds, random_state)

        metric_label = (
            "Error de Clasificación (1 - Accuracy)"
            if task_type == "classification"
            else "Error Cuadrático Medio (RMSE)"
        )

        splits = list(splitter.split(X, y_series))
        max_train_len = min(len(tr) for tr, _ in splits)
        min_sub_size = min(4, max_train_len)

        candidate_sizes = sorted(
            {max(min_sub_size, int(round(max_train_len * frac))) for frac in train_fractions}
        )

        rng = np.random.default_rng(random_state)
        shuffled_splits = [(rng.permutation(tr), val) for tr, val in splits]

        sample_sizes = []
        train_err_mean, train_err_std = [], []
        val_err_mean, val_err_std = [], []

        for n_sub in candidate_sizes:
            fold_tr_errors = []
            fold_val_errors = []

            for shuffled_tr, val_idx in shuffled_splits:
                sub_tr_idx = shuffled_tr[:n_sub]
                X_tr, y_tr = X.iloc[sub_tr_idx], y_series.iloc[sub_tr_idx]
                X_val, y_val = X.iloc[val_idx], y_series.iloc[val_idx]

                if task_type == "classification" and y_tr.nunique() < 2:
                    continue

                if scale is not None:
                    prep = AdaptivePreprocessor()
                    X_tr = prep.fit_transform_train(X_tr, scale=scale)
                    X_val = prep.transform_test(X_val)
                model_clone = copy.deepcopy(model_template)
                model_clone.fit(X_tr, y_tr)

                pred_tr = model_clone.predict(X_tr)
                pred_val = model_clone.predict(X_val)

                if task_type == "classification":
                    fold_tr_errors.append(1.0 - accuracy_score(y_tr, pred_tr))
                    fold_val_errors.append(1.0 - accuracy_score(y_val, pred_val))
                else:
                    fold_tr_errors.append(float(np.sqrt(mean_squared_error(y_tr, pred_tr))))
                    fold_val_errors.append(float(np.sqrt(mean_squared_error(y_val, pred_val))))

            if fold_tr_errors:
                sample_sizes.append(n_sub)
                train_err_mean.append(float(np.mean(fold_tr_errors)))
                train_err_std.append(float(np.std(fold_tr_errors)))
                val_err_mean.append(float(np.mean(fold_val_errors)))
                val_err_std.append(float(np.std(fold_val_errors)))

        return {
            "metric_label": metric_label,
            "sample_sizes": np.array(sample_sizes),
            "train_error_mean": np.array(train_err_mean),
            "train_error_std": np.array(train_err_std),
            "val_error_mean": np.array(val_err_mean),
            "val_error_std": np.array(val_err_std),
        }
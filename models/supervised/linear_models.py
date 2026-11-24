from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression

from ..base import BaseModel


class RegularizedLinearModel(BaseModel):
    """
    Implementación unificada de Modelos Lineales:
    OLS (sin penalización), Ridge (L2), Lasso (L1) y Regresión Logística.
    Soporta extracción de coeficientes ordenados por peso absoluto y alineación de columnas.
    """

    def __init__(self, task_type: str = "regression", penalty: str | None = "l2", **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type.lower()
        self.penalty = penalty.lower() if isinstance(penalty, str) else "none"
        self.feature_names: list[str] | None = None
        self._is_fitted: bool = False

        if self.task_type == "classification":
            sk_penalty = None if self.penalty == "none" else self.penalty
            if sk_penalty not in {None, "l1", "l2"}:
                raise ValueError("En clasificación, penalty debe ser 'l1', 'l2' o 'none'.")
            params = dict(self.hyperparameters)
            params.setdefault("max_iter", 2000)
            params.setdefault("solver", "saga")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = LogisticRegression(penalty=sk_penalty, **params)
        elif self.task_type == "regression":
            if self.penalty == "l2":
                self.model = Ridge(**self.hyperparameters)
            elif self.penalty == "l1":
                params = dict(self.hyperparameters)
                params.setdefault("max_iter", 5000)
                self.model = Lasso(**params)
            elif self.penalty == "none":
                self.model = LinearRegression(**self.hyperparameters)
            else:
                raise ValueError("En regresión, penalty debe ser 'l1', 'l2' o 'none'.")
        else:
            raise ValueError("El parámetro task_type debe ser 'classification' o 'regression'.")

    def _align_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Asegura orden consistente de columnas entre entrenamiento e inferencia."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise ValueError(f"Faltan columnas requeridas para predicción: {missing}")
                return X[self.feature_names]
            return X
        cols = (
            self.feature_names
            if self.feature_names is not None
            else [f"f_{i}" for i in range(X.shape[1])]
        )
        return pd.DataFrame(X, columns=cols)

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> RegularizedLinearModel:
        if y is None:
            raise ValueError("RegularizedLinearModel requiere el vector objetivo 'y' para entrenar.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names = [str(c) for c in X_df.columns]
        X_df = X_df.copy()
        X_df.columns = self.feature_names

        y_arr = np.asarray(y).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError(
                f"El modelo {self.model_name} debe ser entrenado antes de predecir."
            )
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise NotImplementedError(
                "predict_proba solo está disponible para Regresión Logística."
            )
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError(
                f"El modelo {self.model_name} debe ser entrenado antes de predecir."
            )
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)

    def get_coefficients(self) -> pd.DataFrame:
        """Retorna DataFrame ordenado por magnitud absoluta de los coeficientes aprendidos."""
        if not self._is_fitted or self.model is None or not hasattr(self.model, "coef_"):
            raise RuntimeError("El modelo debe ser entrenado antes de extraer coeficientes.")

        coefs = np.asarray(self.model.coef_, dtype=float)
        if coefs.ndim > 1:
            coefs = np.mean(np.abs(coefs), axis=0) if coefs.shape[0] > 1 else coefs[0]

        return (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Coefficient": coefs,
                    "Absolute_Weight": np.abs(coefs),
                }
            )
            .sort_values(by="Absolute_Weight", ascending=False)
            .reset_index(drop=True)
        )
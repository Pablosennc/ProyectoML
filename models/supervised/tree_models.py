from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from ..base import BaseModel


class CARTModel(BaseModel):
    """
    Implementación de Árboles de Decisión CART (Classification and Regression Trees)
    con soporte para poda por costo-complejidad (ccp_alpha), extracción de reglas
    e importancia de características basada en reducción de impureza.
    """

    def __init__(self, task_type: str = "classification", **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type.lower()
        self.feature_names: list[str] | None = None
        self._is_fitted: bool = False

        if self.task_type == "classification":
            self.model = DecisionTreeClassifier(**self.hyperparameters)
        elif self.task_type == "regression":
            self.model = DecisionTreeRegressor(**self.hyperparameters)
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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> CARTModel:
        if y is None:
            raise ValueError("CARTModel requiere el vector objetivo 'y' para entrenar.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names = [str(c) for c in X_df.columns]
        X_df = X_df.copy()
        X_df.columns = self.feature_names

        y_arr = np.asarray(y).ravel()
        self.model.fit(X_df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("CARTModel debe ser entrenado con fit() antes de predecir.")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba solo está disponible para CART en clasificación.")
        if not self._is_fitted or self.model is None:
            raise RuntimeError("CARTModel debe ser entrenado con fit() antes de predecir.")
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna DataFrame ordenado descendentemente con la importancia de cada variable."""
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError(
                "CARTModel debe ser entrenado antes de extraer importancia de variables."
            )

        return (
            pd.DataFrame(
                {
                    "Feature": self.feature_names,
                    "Importance": np.asarray(self.model.feature_importances_, dtype=float),
                }
            )
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )

    def export_rules(self, max_depth: int = 4) -> str:
        """Exporta las reglas de decisión aprendidas en formato textual legible."""
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError("CARTModel debe ser entrenado antes de exportar reglas.")
        return export_text(self.model, feature_names=self.feature_names, max_depth=max_depth)
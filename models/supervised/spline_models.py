from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import SplineTransformer

from ..base import BaseModel

try:
    from pyearth import Earth as _ExternalEarth
    _HAS_PYEARTH = True
except Exception:
    _HAS_PYEARTH = False

try:
    from pygam import LinearGAM as _ExternalLinearGAM, LogisticGAM as _ExternalLogisticGAM
    _HAS_PYGAM = True
except Exception:
    _HAS_PYGAM = False


class _NativeMARSEstimator:
    """
    Implementación nativa en NumPy/SciPy de MARS (Multivariate Adaptive Regression Splines)
    con funciones base bisagra reflejadas max(0, x - t) / max(0, t - x) y selección GCV.
    Garantiza ejecución en Python 3.12+ / 3.14 sin depender de compilaciones Cython heredadas.
    """

    def __init__(self, max_terms: int = 20, max_degree: int = 2, penalty: float = 3.0):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.basis_specs_: list[list[tuple[int, float, int]]] = []
        self.regressor_ = Ridge(alpha=1e-4)

    @staticmethod
    def _eval_term(X: np.ndarray, spec: list[tuple[int, float, int]]) -> np.ndarray:
        col_val = np.ones(X.shape[0], dtype=float)
        for feat_idx, knot, direction in spec:
            if direction == 1:
                col_val *= np.maximum(0.0, X[:, feat_idx] - knot)
            else:
                col_val *= np.maximum(0.0, knot - X[:, feat_idx])
        return col_val

    def _build_design_matrix(
        self, X: np.ndarray, specs: list[list[tuple[int, float, int]]]
    ) -> np.ndarray:
        cols = [np.ones(X.shape[0], dtype=float)]
        for spec in specs:
            cols.append(self._eval_term(X, spec))
        return np.column_stack(cols)

    def _gcv_score(self, B: np.ndarray, y: np.ndarray) -> float:
        n, m = B.shape
        reg = Ridge(alpha=1e-4).fit(B, y)
        rss = float(np.sum((y - reg.predict(B)) ** 2))
        eff_params = m + self.penalty * max(m - 1, 0) / 2.0
        denom = max(1.0 - eff_params / max(n, 1), 1e-3) ** 2
        return (rss / max(n, 1)) / denom

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> _NativeMARSEstimator:
        X_np = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_np.shape

        candidate_knots = {}
        quantiles = np.linspace(0.15, 0.85, min(7, max(2, n_samples // 10)))
        for j in range(n_features):
            uniq = np.unique(X_np[:, j])
            if len(uniq) > 2:
                candidate_knots[j] = np.unique(np.quantile(uniq, quantiles))
            elif len(uniq) == 2:
                candidate_knots[j] = np.array([float(np.mean(uniq))])

        active_specs: list[list[tuple[int, float, int]]] = []
        best_gcv = self._gcv_score(np.ones((n_samples, 1)), y_np)

        while len(active_specs) < self.max_terms:
            best_pair = None
            best_candidate_gcv = best_gcv

            parent_list: list[list[tuple[int, float, int]]] = [[]] + [
                s for s in active_specs if len(s) < self.max_degree
            ]

            for parent in parent_list:
                used_feats = {f for f, _, _ in parent}
                for j, knots in candidate_knots.items():
                    if j in used_feats:
                        continue
                    for t in knots:
                        spec_pos = parent + [(j, float(t), 1)]
                        spec_neg = parent + [(j, float(t), -1)]
                        trial_specs = active_specs + [spec_pos, spec_neg]
                        B_trial = self._build_design_matrix(X_np, trial_specs)
                        gcv = self._gcv_score(B_trial, y_np)
                        if gcv < best_candidate_gcv - 1e-6:
                            best_candidate_gcv = gcv
                            best_pair = [spec_pos, spec_neg]

            if best_pair is None:
                break
            active_specs.extend(best_pair)
            best_gcv = best_candidate_gcv

        self.basis_specs_ = active_specs
        B_final = self._build_design_matrix(X_np, self.basis_specs_)
        self.regressor_.fit(B_final, y_np)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_np = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        B = self._build_design_matrix(X_np, self.basis_specs_)
        return self.regressor_.predict(B)


class SplineModel(BaseModel):
    """
    Implementación de Modelos Aditivos Generalizados (GAM) y
    Splines Adaptativos Multivariados (MARS) con respaldo nativo para Python 3.14+.
    Soporta regresión y clasificación (binaria y multiclase).
    """

    def __init__(self, algorithm: str = "mars", task_type: str = "regression", **kwargs):
        super().__init__(**kwargs)
        self.algorithm = algorithm.lower()
        self.task_type = task_type.lower()
        self.feature_names: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self._is_fitted: bool = False
        self._using_fallback_gam: bool = False
        self._spline_transformer: SplineTransformer | None = None
        self._mars_ovr_models: list = []

        if self.algorithm == "mars":
            self.model = self._create_mars_instance()
        elif self.algorithm == "gam":
            if self.task_type not in {"regression", "classification"}:
                raise ValueError("Para GAM, task_type debe ser 'regression' o 'classification'.")
            if _HAS_PYGAM:
                self.model = (
                    _ExternalLinearGAM(**self.hyperparameters)
                    if self.task_type == "regression"
                    else _ExternalLogisticGAM(**self.hyperparameters)
                )
            else:
                self._init_fallback_gam()
        else:
            raise ValueError("El parámetro algorithm debe ser 'mars' o 'gam'.")

    def _create_mars_instance(self):
        if _HAS_PYEARTH:
            return _ExternalEarth(**self.hyperparameters)
        mars_params = {
            k: v
            for k, v in self.hyperparameters.items()
            if k in {"max_terms", "max_degree", "penalty"}
        }
        return _NativeMARSEstimator(**mars_params)

    def _init_fallback_gam(self) -> None:
        """Inicializa GAM aditivo usando B-Splines cúbicos + regularización Ridge/Logística."""
        self._using_fallback_gam = True
        n_knots = int(self.hyperparameters.get("n_knots", 5))
        self._spline_transformer = SplineTransformer(
            n_knots=n_knots, degree=3, include_bias=False
        )
        if self.task_type == "regression":
            self.model = Ridge(alpha=float(self.hyperparameters.get("alpha", 1.0)))
        else:
            self.model = LogisticRegression(max_iter=2000)

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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> SplineModel:
        if y is None:
            raise ValueError("SplineModel requiere el vector objetivo 'y' para entrenar.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names = [str(c) for c in X_df.columns]
        X_df = X_df.copy()
        X_df.columns = self.feature_names
        y_array = np.asarray(y).ravel()

        if self.task_type == "classification":
            self.classes_ = np.unique(y_array)

        if self.algorithm == "mars" and self.task_type == "classification":
            self._mars_ovr_models = []
            if self.classes_ is not None and len(self.classes_) > 2:
                for cls_val in self.classes_:
                    y_bin = (y_array == cls_val).astype(float)
                    est = self._create_mars_instance().fit(X_df, y_bin)
                    self._mars_ovr_models.append(est)
            else:
                pos_cls = self.classes_[-1] if self.classes_ is not None else 1
                y_bin = (y_array == pos_cls).astype(float)
                self.model.fit(X_df, y_bin)
            self._is_fitted = True
            return self

        if (
            self.algorithm == "gam"
            and self.task_type == "classification"
            and len(self.classes_) > 2
            and not self._using_fallback_gam
        ):
            self._init_fallback_gam()

        if self.algorithm == "gam" and not self._using_fallback_gam:
            try:
                self.model.fit(X_df, y_array)
                self._is_fitted = True
                return self
            except Exception as exc:
                warnings.warn(
                    f"pygam encontró incompatibilidad numérica ({exc}); cambiando a B-Splines nativos."
                )
                self._init_fallback_gam()

        if self._using_fallback_gam and self._spline_transformer is not None:
            X_splines = self._spline_transformer.fit_transform(X_df)
            self.model.fit(X_splines, y_array)
        else:
            self.model.fit(X_df, y_array)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError(
                f"El modelo {self.algorithm.upper()} debe ser entrenado antes de predecir."
            )

        X_aligned = self._align_features(X)

        if self.algorithm == "mars" and self.task_type == "classification":
            probas = self.predict_proba(X_aligned)
            if self.classes_ is not None:
                return self.classes_[np.argmax(probas, axis=1)]
            return (probas[:, 1] >= 0.5).astype(int)

        X_in = (
            self._spline_transformer.transform(X_aligned)
            if self._using_fallback_gam and self._spline_transformer is not None
            else X_aligned
        )
        return self.model.predict(X_in)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba solo está disponible para clasificación.")
        if not self._is_fitted or self.model is None or self.feature_names is None:
            raise RuntimeError(
                f"El modelo {self.algorithm.upper()} debe ser entrenado primero."
            )

        X_aligned = self._align_features(X)

        if self.algorithm == "mars":
            warnings.warn(
                "MARS no genera probabilidades nativas estrictas; retornando salida normalizada en [0, 1]."
            )
            if self._mars_ovr_models:
                raw_scores = np.column_stack(
                    [np.clip(est.predict(X_aligned), 0.0, 1.0) for est in self._mars_ovr_models]
                )
                row_sums = raw_scores.sum(axis=1, keepdims=True)
                return np.divide(
                    raw_scores,
                    row_sums,
                    out=np.full_like(raw_scores, 1.0 / raw_scores.shape[1]),
                    where=row_sums > 0,
                )
            preds = np.clip(self.model.predict(X_aligned), 0.0, 1.0)
            return np.column_stack((1.0 - preds, preds))

        X_in = (
            self._spline_transformer.transform(X_aligned)
            if self._using_fallback_gam and self._spline_transformer is not None
            else X_aligned
        )
        return self.model.predict_proba(X_in)
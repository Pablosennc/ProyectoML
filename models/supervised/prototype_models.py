from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..base import BaseModel


class PrototypeModel(BaseModel):
    """
    Implementación de Métodos de Prototipos y Vecinos:
    - k-NN estándar (clasificación y regresión).
    - LVQ1 (Learning Vector Quantization con actualización competitiva de prototipos).
    - DANN (Discriminant Adaptive Nearest-Neighbor) con métrica elipsoidal local
      y evaluación vectorizada mediante contracción tensorial np.einsum.
    """

    _KNN_ALLOWED_PARAMS = {
        "n_neighbors",
        "weights",
        "algorithm",
        "leaf_size",
        "p",
        "metric",
        "metric_params",
        "n_jobs",
    }

    def __init__(self, algorithm: str = "knn", task_type: str = "classification", **kwargs):
        super().__init__(**kwargs)
        self.algorithm = algorithm.lower()
        self.task_type = task_type.lower()

        self.k_neighbors = int(kwargs.get("n_neighbors", 5))
        self.epsilon = float(kwargs.get("epsilon", 1.0))
        self.lr = float(kwargs.get("learning_rate", 0.05))
        self.epochs = int(kwargs.get("epochs", 15))
        self.random_state = int(kwargs.get("random_state", 42))

        self.feature_names: list[str] | None = None
        self._is_fitted: bool = False

        if self.algorithm == "knn":
            knn_kwargs = {
                k: v for k, v in self.hyperparameters.items() if k in self._KNN_ALLOWED_PARAMS
            }
            knn_kwargs.setdefault("n_neighbors", self.k_neighbors)
            if self.task_type == "classification":
                self.model = KNeighborsClassifier(**knn_kwargs)
            elif self.task_type == "regression":
                self.model = KNeighborsRegressor(**knn_kwargs)
            else:
                raise ValueError("task_type debe ser 'classification' o 'regression'.")
        elif self.algorithm in {"dann", "lvq1"}:
            if self.task_type != "classification":
                raise ValueError(
                    f"{self.algorithm.upper()} solo está implementado para clasificación."
                )
            self.model = self
        else:
            raise ValueError("El algoritmo debe ser 'knn', 'lvq1' o 'dann'.")

        self.X_train_np: np.ndarray | None = None
        self.y_train_np: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.prototypes: dict = {}

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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> PrototypeModel:
        if y is None:
            raise ValueError(f"{self.algorithm.upper()} requiere el vector objetivo 'y' para entrenar.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names = [str(c) for c in X_df.columns]
        X_df = X_df.copy()
        X_df.columns = self.feature_names

        self.X_train_np = X_df.to_numpy(dtype=float)
        self.y_train_np = np.asarray(y).ravel()
        self.classes_ = np.unique(self.y_train_np)

        if len(self.X_train_np) == 0:
            raise ValueError("El conjunto de entrenamiento X no puede estar vacío.")

        if self.algorithm == "knn":
            n_eff = min(max(1, self.k_neighbors), len(self.X_train_np))
            self.model.set_params(n_neighbors=n_eff)
            self.model.fit(X_df, self.y_train_np)

        elif self.algorithm == "lvq1":
            rng = np.random.default_rng(self.random_state)
            self.prototypes = {
                c: self.X_train_np[self.y_train_np == c].mean(axis=0).copy()
                for c in self.classes_
            }
            n_samples = len(self.X_train_np)
            for epoch in range(self.epochs):
                current_lr = self.lr * (1.0 - epoch / max(self.epochs, 1))
                perm = rng.permutation(n_samples)
                for idx in perm:
                    x_i = self.X_train_np[idx]
                    y_i = self.y_train_np[idx]
                    dists = {
                        c: float(np.linalg.norm(x_i - p)) for c, p in self.prototypes.items()
                    }
                    closest_c = min(dists, key=dists.get)
                    if closest_c == y_i:
                        self.prototypes[closest_c] += current_lr * (
                            x_i - self.prototypes[closest_c]
                        )
                    else:
                        self.prototypes[closest_c] -= current_lr * (
                            x_i - self.prototypes[closest_c]
                        )

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.X_train_np is None or self.y_train_np is None:
            raise RuntimeError(
                f"El modelo {self.algorithm.upper()} debe entrenarse con fit() antes de predecir."
            )

        X_aligned = self._align_features(X)

        if self.algorithm == "knn":
            return self.model.predict(X_aligned)

        X_test_np = X_aligned.to_numpy(dtype=float)
        if self.algorithm == "lvq1":
            preds = []
            for x_i in X_test_np:
                dists = {c: float(np.linalg.norm(x_i - p)) for c, p in self.prototypes.items()}
                preds.append(min(dists, key=dists.get))
            return np.array(preds)

        return self._predict_dann(X_test_np)

    def _predict_dann(self, X_test: np.ndarray) -> np.ndarray:
        """
        Calcula la métrica local discriminante Sigma(x0) en una vecindad de tamaño k0
        y evalúa distancias elipsoidales mediante contracción tensorial np.einsum.
        """
        n_train, p = self.X_train_np.shape
        k_eff = min(max(1, self.k_neighbors), n_train)
        k0 = min(n_train, max(50, k_eff * 5))
        eye_p = np.eye(p, dtype=float)
        preds = []

        for x_0 in X_test:
            diff_eucl = self.X_train_np - x_0
            dist_eucl = np.linalg.norm(diff_eucl, axis=1)
            idx_k0 = np.argsort(dist_eucl)[:k0]

            X_local = self.X_train_np[idx_k0]
            y_local = self.y_train_np[idx_k0]

            W = np.zeros((p, p), dtype=float)
            B = np.zeros((p, p), dtype=float)
            x_mean = np.mean(X_local, axis=0)

            for c in np.unique(y_local):
                X_c = X_local[y_local == c]
                pi_k = len(X_c) / k0
                if len(X_c) > 1:
                    cov_c = np.atleast_2d(np.cov(X_c, rowvar=False, bias=True))
                else:
                    cov_c = np.zeros((p, p), dtype=float)
                W += pi_k * cov_c

                mean_c = np.mean(X_c, axis=0)
                mean_diff = (mean_c - x_mean).reshape(-1, 1)
                B += pi_k * (mean_diff @ mean_diff.T)

            W_reg = np.linalg.pinv(W + 1e-6 * eye_p)
            W_inv_half = np.real(fractional_matrix_power(W_reg, 0.5))
            B_star = W_inv_half @ B @ W_inv_half
            Sigma_x0 = W_inv_half @ (B_star + self.epsilon * eye_p) @ W_inv_half

            # Contracción tensorial vectorizada: (N, p) x (p, p) x (N, p) -> (N,)
            diff = self.X_train_np - x_0
            dist_dann = np.einsum("ij,jk,ik->i", diff, Sigma_x0, diff)

            idx_k = np.argsort(dist_dann)[:k_eff]
            k_nearest_labels = self.y_train_np[idx_k]
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba solo está disponible para clasificación.")
        if not self._is_fitted:
            raise RuntimeError(
                f"El modelo {self.algorithm.upper()} debe entrenarse con fit() antes de predecir."
            )
        if self.algorithm == "knn":
            X_aligned = self._align_features(X)
            return self.model.predict_proba(X_aligned)
        raise NotImplementedError(
            f"predict_proba no está soportado nativamente en {self.algorithm.upper()}."
        )
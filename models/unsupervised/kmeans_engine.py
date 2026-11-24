from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..base import BaseModel


class KMeansEngine(BaseModel):
    """
    Motor de aprendizaje no supervisado basado en K-Means / K-Means++.
    Ejecuta auditoría completa de particionamiento, detección formal del codo
    (máxima distancia perpendicular a la secante normalizada), análisis de siluetas
    individuales/globales, índices Calinski-Harabasz y Davies-Bouldin, y proyección PCA 2D.
    """

    _KMEANS_ALLOWED_PARAMS = {
        "max_iter",
        "tol",
        "verbose",
        "copy_x",
        "algorithm",
    }

    def __init__(
        self,
        k_range: tuple[int, int] = (2, 10),
        selection_criterion: str = "elbow",
        init: str = "k-means++",
        n_init: int = 10,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        k_min, k_max = int(k_range[0]), int(k_range[1])
        if k_min < 2 or k_max <= k_min:
            raise ValueError("k_range debe cumplir 2 <= k_min < k_max.")
        if selection_criterion.lower() not in {"elbow", "silhouette"}:
            raise ValueError("selection_criterion debe ser 'elbow' o 'silhouette'.")

        self.k_range = (k_min, k_max)
        self.selection_criterion = selection_criterion.lower()
        self.init = init
        self.n_init = n_init
        self.random_state = random_state

        self.feature_names: list[str] | None = None
        self._is_fitted: bool = False

        self.audit_results_: pd.DataFrame | None = None
        self.optimal_k_: int | None = None
        self.labels_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.sample_silhouettes_: np.ndarray | None = None
        self.avg_silhouette_: float | None = None
        self.pca_: PCA | None = None
        self.X_pca_: np.ndarray | None = None
        self.centroids_pca_: np.ndarray | None = None

    @staticmethod
    def _detect_formal_elbow(k_values: np.ndarray, inertias: np.ndarray) -> int:
        """
        Detecta formalmente el punto de codo mediante el método geométrico de
        máxima distancia perpendicular a la recta secante en el plano normalizado [0, 1]^2.
        """
        if len(k_values) <= 2:
            return int(k_values[0])

        k_span = float(k_values.max() - k_values.min())
        i_span = float(inertias.max() - inertias.min())
        if k_span == 0.0 or i_span == 0.0:
            return int(k_values[0])

        k_norm = (k_values - k_values.min()) / k_span
        i_norm = (inertias - inertias.min()) / i_span

        p1 = np.array([k_norm[0], i_norm[0]])
        p2 = np.array([k_norm[-1], i_norm[-1]])
        line_vec = p2 - p1
        line_norm = float(np.linalg.norm(line_vec))

        if line_norm == 0.0:
            return int(k_values[0])

        points = np.column_stack((k_norm, i_norm))
        vec_from_p1 = points - p1
        cross_prod = np.abs(
            line_vec[0] * vec_from_p1[:, 1] - line_vec[1] * vec_from_p1[:, 0]
        )
        distances = cross_prod / line_norm

        best_idx = int(np.argmax(distances))
        return int(k_values[best_idx])

    def _align_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Asegura orden consistente de columnas y conversión limpia a arreglo NumPy."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise ValueError(f"Faltan columnas requeridas para predicción: {missing}")
                X_sub = X[self.feature_names]
            else:
                X_sub = X
            numeric_X = X_sub.select_dtypes(include=[np.number])
            if numeric_X.shape[1] != X_sub.shape[1]:
                raise ValueError(
                    "KMeansEngine requiere que todas las columnas de X sean numéricas y estén escaladas."
                )
            return numeric_X.to_numpy(dtype=float)

        arr = np.asarray(X, dtype=float)
        if self.feature_names is not None and arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Se esperaban {len(self.feature_names)} columnas, pero se recibieron {arr.shape[1]}."
            )
        return arr

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> KMeansEngine:
        """
        Ejecuta el barrido de K, selecciona el K óptimo, ajusta el modelo final
        y proyecta observaciones y centroides en 2D mediante PCA.
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names = [str(c) for c in X_df.columns]
        X_df = X_df.copy()
        X_df.columns = self.feature_names

        X_np = self._align_features(X_df)
        if np.isnan(X_np).any() or np.isinf(X_np).any():
            raise ValueError("KMeansEngine no admite valores NaN ni infinitos en X.")

        n_samples = X_np.shape[0]
        k_min, k_max = self.k_range

        if n_samples <= k_min:
            raise ValueError(
                f"El número de muestras ({n_samples}) debe ser mayor que k_min ({k_min})."
            )

        effective_k_max = min(k_max, n_samples - 1)
        if effective_k_max < k_max:
            warnings.warn(
                f"Dataset pequeño (n={n_samples}): se ajusta k_max automáticamente de {k_max} a {effective_k_max}."
            )

        km_extra = {
            k: v for k, v in self.hyperparameters.items() if k in self._KMEANS_ALLOWED_PARAMS
        }

        records = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(k_min, effective_k_max + 1):
                km = KMeans(
                    n_clusters=k,
                    init=self.init,
                    n_init=self.n_init,
                    random_state=self.random_state,
                    **km_extra,
                )
                labels = km.fit_predict(X_np)

                # Si hay filas duplicadas y colapsan en menos de 2 clústeres, detener barrido
                if len(np.unique(labels)) < 2:
                    break

                records.append(
                    {
                        "k": k,
                        "inertia": float(km.inertia_),
                        "silhouette": float(silhouette_score(X_np, labels)),
                        "calinski_harabasz": float(calinski_harabasz_score(X_np, labels)),
                        "davies_bouldin": float(davies_bouldin_score(X_np, labels)),
                    }
                )

        if not records:
            raise ValueError(
                "No fue posible formar al menos 2 clústeres distintos (verifique que los datos tengan varianza)."
            )

        self.audit_results_ = pd.DataFrame(records)

        if self.selection_criterion == "elbow":
            self.optimal_k_ = self._detect_formal_elbow(
                self.audit_results_["k"].to_numpy(dtype=float),
                self.audit_results_["inertia"].to_numpy(dtype=float),
            )
        else:
            best_row = self.audit_results_.loc[
                self.audit_results_["silhouette"].idxmax()
            ]
            self.optimal_k_ = int(best_row["k"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = KMeans(
                n_clusters=self.optimal_k_,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
                **km_extra,
            )
            self.labels_ = self.model.fit_predict(X_np)

        self.centroids_ = self.model.cluster_centers_
        self.sample_silhouettes_ = silhouette_samples(X_np, self.labels_)
        self.avg_silhouette_ = float(np.mean(self.sample_silhouettes_))

        n_components = min(2, X_np.shape[1], n_samples)
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.X_pca_ = self.pca_.fit_transform(X_np)
        self.centroids_pca_ = self.pca_.transform(self.centroids_)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Asigna cada nueva observación al centroide más cercano en distancia euclidiana."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("KMeansEngine debe ajustarse con fit() antes de predecir.")
        X_np = self._align_features(X)
        return self.model.predict(X_np)

    def score(self, X: pd.DataFrame, y=None) -> float:
        """Retorna el negativo de la inercia del modelo sobre X."""
        return float(self.model.score(self._align_features(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """K-Means realiza particionamiento duro y no estima probabilidades posteriores nativas."""
        raise NotImplementedError(
            "KMeansEngine ejecuta particionamiento duro (hard clustering) y no soporta predict_proba."
        )

    def get_audit_report(self) -> dict:
        """Empaqueta todos los artefactos numéricos y proyecciones para HDVisualizer."""
        if not self._is_fitted or self.model is None or self.audit_results_ is None:
            raise RuntimeError("Debe ejecutar fit() antes de solicitar el reporte de auditoría.")

        return {
            "optimal_k": self.optimal_k_,
            "selection_criterion": self.selection_criterion,
            "metrics_table": self.audit_results_,
            "labels": self.labels_,
            "centroids": self.centroids_,
            "sample_silhouettes": self.sample_silhouettes_,
            "avg_silhouette": self.avg_silhouette_,
            "X_pca": self.X_pca_,
            "centroids_pca": self.centroids_pca_,
            "pca_explained_variance": self.pca_.explained_variance_ratio_,
        }
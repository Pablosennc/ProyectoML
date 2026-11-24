"""Motor e-means con la implementación de CPU publicada por Meta FAIR."""

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from .kmeans_engine import KMeansEngine


class _FaissEMeansEstimator:
    """Adapta Faiss a la auditoría y conserva los centroides para inferencia."""

    def __init__(self, n_clusters: int, seed: int):
        self.n_clusters = n_clusters
        self.seed = seed

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        try:
            from faiss.contrib.e_means import NumpyEmeans
        except ImportError as exc:
            raise ImportError(
                "e-means requiere Faiss. Ejecuta python -m pip install -r requirements-emeans.txt."
            ) from exc

        vectors = np.ascontiguousarray(X, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Las variables del modelo deben ser representables en float32.")
        trainer = NumpyEmeans(vectors.shape[1], self.n_clusters, seed=self.seed)
        centers, _ = trainer.train(vectors)
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        labels, distances = self._assign(X)
        self.inertia_ = float(np.square(distances).sum())
        return labels

    def _assign(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pairwise_distances_argmin_min(X, self.cluster_centers_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X)[0]

    def score(self, X: np.ndarray) -> float:
        _, distances = self._assign(X)
        return -float(np.square(distances).sum())


class EMeansEngine(KMeansEngine):
    """Reutiliza el barrido de K y la auditoría con el estimador e-means."""

    algorithm_name = "e-means"
    report_prefix = "emeans"

    def _make_estimator(self, k: int) -> _FaissEMeansEstimator:
        return _FaissEMeansEstimator(k, seed=self.random_state)

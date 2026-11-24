from __future__ import annotations
import numpy as np
import pandas as pd


class AutomatedEDA:
    """
    Motor de análisis exploratorio automatizado (EDA).
    Calcula diagnósticos de calidad, multicolinealidad, outliers por IQR y matrices
    de covarianza, Pearson y Spearman tolerando columnas constantes, infinitos o nulos.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_quality_diagnostics(self) -> dict:
        """Retorna dimensiones, filas duplicadas, conteo de nulos por columna y tipos de datos."""
        missing = self.df.isnull().sum()
        missing_dict = {str(k): int(v) for k, v in missing[missing > 0].items()}
        return {
            "shape": self.df.shape,
            "duplicate_rows": int(self.df.duplicated().sum()),
            "missing_values": missing_dict,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
        }

    def _get_valid_numeric_df(self, max_cols: int | None = None) -> pd.DataFrame:
        """Filtra columnas numéricas (excluyendo booleanos) con varianza estrictamente positiva."""
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        num_df = num_df.replace([np.inf, -np.inf], np.nan)
        valid_cols = [c for c in num_df.columns if num_df[c].nunique(dropna=True) > 1]
        num_df = num_df[valid_cols]

        if max_cols is not None and num_df.shape[1] > max_cols:
            top_cols = num_df.var(ddof=1).sort_values(ascending=False).head(max_cols).index
            num_df = num_df[top_cols]

        return num_df

    def detect_multicollinearity(self, threshold: float = 0.85) -> list[tuple[str, str, float]]:
        """Detecta pares de variables numéricas con correlación lineal |r| >= threshold."""
        num_df = self._get_valid_numeric_df()
        if num_df.shape[1] < 2:
            return []

        corr_matrix = num_df.corr(method="pearson").abs()
        pairs: list[tuple[str, str, float]] = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr_matrix.iloc[i, j])
                if not np.isnan(val) and val >= threshold:
                    pairs.append((str(cols[i]), str(cols[j]), round(val, 4)))
        return pairs

    def detect_outliers_iqr(self, factor: float = 1.5) -> dict[str, int]:
        """Cuantifica observaciones atípicas por variable continua usando rango intercuartílico."""
        num_df = self._get_valid_numeric_df()
        outlier_counts: dict[str, int] = {}

        for col in num_df.columns:
            series = num_df[col].dropna()
            if series.nunique() <= 2:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0.0:
                continue
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            count = int(((series < lower) | (series > upper)).sum())
            if count > 0:
                outlier_counts[str(col)] = count

        return outlier_counts

    def calculate_matrices(
        self, max_cols: int = 25
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calcula matrices de Covarianza, Correlación de Pearson y Correlación de Spearman.
        Limita a las top `max_cols` variables por varianza si el dataset tiene muchas columnas.
        """
        num_df = self._get_valid_numeric_df(max_cols=max_cols)
        if num_df.shape[1] < 2:
            raise ValueError(
                "Se requieren al menos 2 columnas numéricas con varianza para calcular matrices."
            )

        cov_matrix = num_df.cov().fillna(0.0)
        pearson_matrix = num_df.corr(method="pearson").fillna(0.0)
        spearman_matrix = num_df.corr(method="spearman").fillna(0.0)

        return cov_matrix, pearson_matrix, spearman_matrix
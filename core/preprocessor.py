from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AdaptivePreprocessor:
    """
    Preprocesador adaptativo libre de fuga de datos (Data Leakage).
    - Sanitiza automáticamente IDs, columnas constantes y texto de alta cardinalidad.
    - Ajusta imputadores, escaladores Z-score y One-Hot Encoding estrictamente en Train.
    - Replica las transformaciones exactamente alineadas en Test.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        max_cardinality: int = 20,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.max_cardinality = max_cardinality

        self.scale: bool = True
        self.selected_features_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.bool_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.encoded_columns_: list[str] = []

        self.num_imputer: SimpleImputer | None = None
        self.cat_imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None

    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en entrenamiento y prueba descartando filas con target nulo."""
        if self.target_col is None or self.target_col not in self.df.columns:
            raise ValueError("Debe especificar un target_col válido para dividir los datos.")

        clean_df = self.df.dropna(subset=[self.target_col]).reset_index(drop=True)
        X = clean_df.drop(columns=[self.target_col])
        y = clean_df[self.target_col]

        stratify_arg = None
        if stratify:
            class_counts = y.value_counts()
            if len(class_counts) >= 2 and class_counts.min() >= 2:
                stratify_arg = y
            else:
                warnings.warn(
                    "Alguna clase tiene menos de 2 muestras; se omite estratificación en train_test_split."
                )

        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

    def _filter_informative_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Descarta columnas vacías, constantes, IDs únicos y texto de alta cardinalidad."""
        n_rows = max(len(X), 1)
        keep_cols = []
        dropped_cols = []

        for col in X.columns:
            series = X[col].replace([np.inf, -np.inf], np.nan)
            n_unique = series.nunique(dropna=True)

            if n_unique <= 1:
                dropped_cols.append(col)
                continue

            col_lower = str(col).strip().lower()
            is_id_name = col_lower in {"id", "index", "unnamed: 0", "customer_id", "user_id"}
            if is_id_name and n_unique >= 0.95 * n_rows:
                dropped_cols.append(col)
                continue

            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                if n_unique > self.max_cardinality or (n_unique / n_rows) > 0.5:
                    dropped_cols.append(col)
                    continue

            keep_cols.append(col)

        if dropped_cols:
            print(f"[PREPROCESSOR] Columnas no informativas descartadas: {dropped_cols}")

        if not keep_cols:
            raise ValueError("No quedaron columnas predictoras válidas tras el filtrado automático.")

        return X[keep_cols].copy()

    def fit_transform_train(self, X_train: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """Ajusta imputadores, escalador y codificación sobre X_train y lo transforma."""
        self.scale = scale
        X_clean = self._filter_informative_columns(X_train)
        self.selected_features_ = X_clean.columns.tolist()

        self.bool_cols_ = X_clean.select_dtypes(include=["bool"]).columns.tolist()
        self.numeric_cols_ = [
            c
            for c in X_clean.select_dtypes(include=[np.number]).columns
            if c not in self.bool_cols_
        ]
        self.categorical_cols_ = [
            c
            for c in self.selected_features_
            if c not in self.numeric_cols_ and c not in self.bool_cols_
        ]

        blocks = []

        num_and_bool_cols = self.numeric_cols_ + self.bool_cols_
        if num_and_bool_cols:
            num_data = X_clean[num_and_bool_cols].astype(float).replace([np.inf, -np.inf], np.nan)
            self.num_imputer = SimpleImputer(strategy="median")
            num_arr = self.num_imputer.fit_transform(num_data)

            if self.scale:
                self.scaler = StandardScaler()
                num_arr = self.scaler.fit_transform(num_arr)

            blocks.append(pd.DataFrame(num_arr, columns=num_and_bool_cols, index=X_clean.index))

        if self.categorical_cols_:
            cat_data = X_clean[self.categorical_cols_].astype(object)
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_arr = self.cat_imputer.fit_transform(cat_data)
            cat_df = pd.DataFrame(cat_arr, columns=self.categorical_cols_, index=X_clean.index)
            cat_encoded = pd.get_dummies(
                cat_df, columns=self.categorical_cols_, drop_first=True, dtype=float
            )
            blocks.append(cat_encoded)

        X_out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=X_clean.index)
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.encoded_columns_ = X_out.columns.tolist()
        return X_out

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Aplica sobre X_test los parámetros aprendidos en X_train sin fuga de datos."""
        if not self.selected_features_:
            raise RuntimeError("Debe ejecutar fit_transform_train() antes de transform_test().")

        X_sub = X_test[self.selected_features_].copy()
        blocks = []

        num_and_bool_cols = self.numeric_cols_ + self.bool_cols_
        if num_and_bool_cols and self.num_imputer is not None:
            num_data = X_sub[num_and_bool_cols].astype(float).replace([np.inf, -np.inf], np.nan)
            num_arr = self.num_imputer.transform(num_data)
            if self.scale and self.scaler is not None:
                num_arr = self.scaler.transform(num_arr)
            blocks.append(pd.DataFrame(num_arr, columns=num_and_bool_cols, index=X_sub.index))

        if self.categorical_cols_ and self.cat_imputer is not None:
            cat_data = X_sub[self.categorical_cols_].astype(object)
            cat_arr = self.cat_imputer.transform(cat_data)
            cat_df = pd.DataFrame(cat_arr, columns=self.categorical_cols_, index=X_sub.index)
            cat_encoded = pd.get_dummies(
                cat_df, columns=self.categorical_cols_, drop_first=True, dtype=float
            )
            blocks.append(cat_encoded)

        X_out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=X_sub.index)
        X_out = X_out.reindex(columns=self.encoded_columns_, fill_value=0.0)
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X_out
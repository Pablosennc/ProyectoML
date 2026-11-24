from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd


class DataLoader:
    """
    Módulo de ingesta agnóstica de datos tabulares.
    Detecta automáticamente formato, codificación, delimitador, decimales con coma,
    columna objetivo y tipo de tarea (clasificación o regresión).
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {self.filepath}")
        self.df: pd.DataFrame | None = None

    def _read_delimited_file(self) -> pd.DataFrame:
        """Detecta automáticamente codificación, separador (',', ';', '\\t', '|') y coma decimal."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        candidate_seps = [",", ";", "\t", "|"]

        for enc in encodings:
            try:
                with open(self.filepath, "r", encoding=enc, errors="strict") as f:
                    sample = f.read(4096)

                detected_sep = None
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidate_seps))
                    detected_sep = dialect.delimiter
                except csv.Error:
                    detected_sep = None

                seps_to_try = (
                    [detected_sep] + [s for s in candidate_seps if s != detected_sep]
                    if detected_sep
                    else candidate_seps
                )

                for sep in seps_to_try:
                    df = pd.read_csv(self.filepath, sep=sep, encoding=enc, low_memory=False)
                    if df.shape[1] > 1:
                        # Si el separador es ';' y hay columnas numéricas con coma decimal (ej. '12,5')
                        if sep == ";":
                            obj_cols = df.select_dtypes(include=["object", "string"]).columns
                            has_comma_decimals = any(
                                df[c].dropna().astype(str).str.match(r"^-?\d+,\d+$").mean() > 0.8
                                for c in obj_cols
                                if not df[c].dropna().empty
                            )
                            if has_comma_decimals:
                                df = pd.read_csv(
                                    self.filepath,
                                    sep=sep,
                                    encoding=enc,
                                    decimal=",",
                                    low_memory=False,
                                )
                        return df

                return pd.read_csv(self.filepath, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"No fue posible decodificar el archivo '{self.filepath}' con codificaciones estándar.")

    def load_data(self) -> pd.DataFrame:
        """Carga CSV, TSV, TXT, Parquet o Excel limpiando encabezados y filas/columnas vacías."""
        suffix = self.filepath.suffix.lower()

        if suffix in {".csv", ".tsv", ".txt", ".data"}:
            df = self._read_delimited_file()
        elif suffix == ".parquet":
            df = pd.read_parquet(self.filepath)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(self.filepath)
        else:
            df = self._read_delimited_file()

        df.columns = [str(col).strip() for col in df.columns]
        df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

        if df.empty or df.shape[1] < 1:
            raise ValueError("El dataset cargado está vacío.")

        if not df.columns.is_unique:
            raise ValueError("Los nombres de columna deben ser únicos.")

        self.df = df
        return df

    def detect_task_and_target(
        self, target_col: str | None = None, df: pd.DataFrame | None = None
    ) -> tuple[str | None, str | None]:
        """Detecta el objetivo en una tabla explícita o previamente cargada."""
        data = df if df is not None else self.df
        if data is None:
            raise RuntimeError("Debe ejecutar load_data() antes de detectar tarea y objetivo.")
        return self.resolve_target(data, target_col)

    @staticmethod
    def resolve_target(
        df: pd.DataFrame, target_col: str | None = None
    ) -> tuple[str | None, str | None]:
        """Detecta un objetivo por nombre semántico o acepta uno explícito."""
        resolved_target = target_col
        if resolved_target is None:
            aliases = {"target", "label", "class", "clase", "objetivo", "etiqueta", "y"}
            candidates = [c for c in df.columns if c.casefold() in aliases
                          and df[c].nunique(dropna=True) > 1]
            if len(candidates) != 1:
                return None, None
            resolved_target = candidates[0]
        data = df
        if resolved_target not in data.columns:
            raise ValueError(
                f"La columna objetivo '{resolved_target}' no existe. "
                f"Columnas disponibles: {data.columns.tolist()}"
            )

        y = data[resolved_target].dropna()
        if y.empty:
            raise ValueError(f"La columna objetivo '{resolved_target}' contiene únicamente valores nulos.")

        n_unique = y.nunique()
        n_samples = len(y)

        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            task_type = "classification"
        else:
            is_integer_like = np.all(np.isclose(y.to_numpy(dtype=float) % 1, 0))
            if is_integer_like and n_unique <= 20 and n_unique < (n_samples * 0.5):
                task_type = "classification"
            else:
                task_type = "regression"

        return resolved_target, task_type
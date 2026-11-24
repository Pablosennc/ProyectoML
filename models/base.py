from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos del framework (supervisados y no supervisados).
    Define el contrato polimórfico obligatorio (fit, predict, predict_proba, save, load)
    y encapsula la persistencia en disco mediante joblib.
    """

    def __init__(self, **kwargs):
        self.model = None
        self.model_name: str = self.__class__.__name__
        self.hyperparameters: dict = dict(kwargs)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> BaseModel:
        """Ajusta el modelo a los datos de entrenamiento."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones o asignaciones de clúster sobre nuevos datos."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Genera probabilidades posteriores por clase (si el algoritmo lo soporta)."""
        pass

    def save(self, filepath: str | Path) -> Path:
        """Serializa la instancia completa del modelo en disco creando directorios padre."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, filepath: str | Path) -> BaseModel:
        """Carga desde disco un modelo serializado y valida su contrato BaseModel."""
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de modelo en: {in_path}")
        loaded_obj = joblib.load(in_path)
        if not isinstance(loaded_obj, BaseModel):
            raise TypeError(
                f"El archivo '{in_path}' no contiene una instancia válida de BaseModel."
            )
        return loaded_obj
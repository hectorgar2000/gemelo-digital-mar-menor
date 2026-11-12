"""Modelo de SST que sirve el gemelo: árboles (XGBoost) y, opcionalmente, una parte lineal promediada con ellos.

    predicción = (1 - peso) · árboles(X) + peso · lineal(X)

La parte lineal (ridge sobre variables estandarizadas) se guarda como un diccionario de números (medias, escalas,
coeficientes): es JSON portable, sin pickle, y se puede leer a simple vista. Con peso = 0 (o sin parte lineal)
el modelo es un XGBoost normal, así que los modelos antiguos se cargan igual.
"""
import numpy as np
import pandas as pd
import xgboost as xgb


class ModeloSST:
    def __init__(self, arboles: xgb.XGBRegressor, lineal: dict | None = None):
        self.arboles = arboles
        self.lineal = lineal
        self._explicador = None

    # ------------------------------------------------------------------ construcción / guardado
    @classmethod
    def cargar(cls, ruta_json, lineal: dict | None = None) -> "ModeloSST":
        m = xgb.XGBRegressor()
        m.load_model(ruta_json)
        return cls(m, lineal)

    @staticmethod
    def lineal_desde_sklearn(pipeline, variables, peso: float) -> dict:
        """De un Pipeline(StandardScaler, Ridge) ya entrenado al diccionario portable."""
        escalador, ridge = pipeline[0], pipeline[-1]
        return {"peso": float(peso), "variables": list(variables), "media": escalador.mean_.tolist(),
                "escala": escalador.scale_.tolist(), "coef": ridge.coef_.tolist(), "intercepto": float(ridge.intercept_)}

    # ------------------------------------------------------------------ uso
    @property
    def variables(self) -> list[str]:
        return self.arboles.get_booster().feature_names

    @property
    def peso_lineal(self) -> float:
        return self.lineal["peso"] if self.lineal else 0.0

    def get_booster(self):
        return self.arboles.get_booster()

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        li = self.lineal
        return (X[li["variables"]].to_numpy(dtype=float) - np.array(li["media"])) / np.array(li["escala"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.arboles.predict(X[self.variables])
        if not self.lineal:
            return p
        w = self.peso_lineal
        return (1 - w) * p + w * (self._z(X) @ np.array(self.lineal["coef"]) + self.lineal["intercepto"])

    def contribuciones(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Contribución de cada variable a la predicción de cada fila (n_filas × n_variables) y valor base.
        valor base + suma de contribuciones = predicción (aditivo, como SHAP).
          - árboles: valores SHAP (TreeExplainer);
          - lineal: coeficiente × valor estandarizado (exacto para un modelo lineal; base = intercepto)."""
        import shap
        if self._explicador is None:
            self._explicador = shap.TreeExplainer(self.arboles)
        sv = self._explicador(X[self.variables])
        valores, base = np.asarray(sv.values, dtype=float), float(np.ravel(sv.base_values)[0])
        if not self.lineal:
            return valores, base
        w = self.peso_lineal
        lin = pd.DataFrame(self._z(X) * np.array(self.lineal["coef"]), columns=self.lineal["variables"])
        valores = (1 - w) * valores + w * lin[self.variables].to_numpy()
        return valores, (1 - w) * base + w * self.lineal["intercepto"]

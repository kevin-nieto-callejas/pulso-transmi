"""Comprueba que el champion vigente se puede CARGAR Y USAR en un entorno limpio.

Existe por un riesgo concreto: el champion paso a ser un ensamble de tres
familias de boosting (LightGBM + XGBoost + CatBoost). joblib necesita esas
mismas librerias instaladas para reconstruirlo. Si alguna falta en el runner
de GitHub Actions, la inferencia fallaria... pero solo el dia que haya un
ciclo abierto, porque `infer.py` termina antes de cargar el modelo cuando no
hay ciclo. Es decir: el fallo estaria escondido hasta el peor momento posible.

Esta comprobacion no depende de que exista un ciclo: descarga el artefacto,
lo reconstruye y le pide una prediccion sobre una fila sintetica. Si el
entorno no puede con el modelo, se entera aqui y no en plena competencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import get_champion, load_model_from_storage  # noqa: E402


def main() -> None:
    champion = get_champion()
    print(f"Champion: {champion['version_id']} (metrica={champion['validation_metric']:.2f})")
    print(f"Artefacto: {champion['artifact_location']}")

    bundle = load_model_from_storage(champion["artifact_location"])
    model, columnas = bundle["model"], bundle["feature_columns"]
    print(f"Modelo reconstruido: {type(model).__name__} con {len(columnas)} features")

    if hasattr(model, "estimators_") or hasattr(model, "estimators"):
        partes = [type(e).__name__ for _, e in getattr(model, "estimators", [])]
        if partes:
            print(f"  Ensamble de: {', '.join(partes)}")

    # Fila sintetica: valores neutros. No interesa el numero, interesa que el
    # modelo sea capaz de producirlo sin reventar por una libreria ausente.
    fila = pd.DataFrame([{c: 0.0 for c in columnas}])
    prediccion = float(model.predict(fila)[0])

    if not np.isfinite(prediccion):
        raise RuntimeError(f"El modelo devolvio un valor no finito: {prediccion}")

    print(f"Prediccion de prueba: {prediccion:.2f}")
    print("OK: el champion se carga y predice en este entorno.")


if __name__ == "__main__":
    main()

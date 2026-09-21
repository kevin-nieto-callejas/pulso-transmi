"""Revalida los mejores del barrido con el protocolo oficial.

El barrido (`src/sweep.py`) explora en volumen con validacion barata: 3
cortes temporales, para poder recorrer miles de combinaciones. Esos numeros
NO son comparables con el champion, que se midio con 5 cortes. Compararlos
seria repetir el error que ya costo caro en este proyecto: contrastar cifras
calculadas de formas distintas.

Este script cierra el ciclo. Toma los mejores candidatos del barrido, los
vuelve a medir con el protocolo completo, y dice cuales superarian de verdad
al champion. No promueve nada: eso sigue siendo decision de `train.py`.

    python src/revalidar.py            # top 10
    python src/revalidar.py --top 25
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_feature_frame, explode_horizons, station_dummy_columns  # noqa: E402
from sweep import conjuntos_de_features, construir_modelo  # noqa: E402
from train import evaluate_candidate, load_data  # noqa: E402

SEED = 20260918
RESULTADOS = ROOT / "eda" / "reports" / "sweep_results.csv"


def champion_actual() -> tuple[str, float] | None:
    url = os.environ.get("SUPABASE_URL")
    clave = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not clave:
        return None
    r = httpx.get(
        f"{url.rstrip('/')}/rest/v1/model_versions",
        headers={"apikey": clave, "Authorization": f"Bearer {clave}"},
        params={"status": "eq.champion", "select": "version_id,validation_metric"},
        timeout=30.0,
    )
    r.raise_for_status()
    filas = r.json()
    return (filas[0]["version_id"], filas[0]["validation_metric"]) if filas else None


def params_del_intento(fila: pd.Series, familia: str) -> dict:
    """Reconstruye los hiperparametros desde la fila del CSV.

    El CSV guarda una columna por parametro de TODAS las familias, asi que
    las que no aplican vienen vacias.
    """
    ignorar = {"trial", "familia", "feature_set", "accuracy"}
    return {
        k: (int(v) if float(v).is_integer() else float(v))
        for k, v in fila.items()
        if k not in ignorar and pd.notna(v) and v != ""
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Revalida los mejores del barrido")
    parser.add_argument("--top", type=int, default=10, help="cuantos revalidar")
    args = parser.parse_args()

    if not RESULTADOS.exists():
        raise SystemExit(f"No existe {RESULTADOS}. Corre primero src/sweep.py")

    tabla = pd.read_csv(RESULTADOS).sort_values("accuracy", ascending=False)
    if "familia" not in tabla.columns:
        tabla["familia"] = "xgboost"  # barridos anteriores solo usaban XGBoost
    mejores = tabla.head(args.top)

    observations, context = load_data()
    feature_frame = explode_horizons(build_feature_frame(observations, context))
    variantes = conjuntos_de_features(observations)
    extendido = station_dummy_columns(observations)

    champion = champion_actual()
    if champion:
        print(f"Champion actual: {champion[0]} ({champion[1]:.2f})\n")
    else:
        print("Sin credenciales de Supabase: se revalida igual, sin comparar.\n")

    print(f"Revalidando los {len(mejores)} mejores con 5 cortes (el protocolo del champion):\n")
    print("  trial  familia    features          barrido  oficial   vs champion")
    filas = []
    for _, fila in mejores.iterrows():
        familia = fila["familia"]
        columnas = variantes.get(fila["feature_set"], variantes["con_estacion"])
        params = params_del_intento(fila, familia)
        constructor = lambda **_: construir_modelo(familia, params, SEED)  # noqa: E731

        try:
            oficial = evaluate_candidate(constructor, columnas, {}, feature_frame, n_splits=5)
        except Exception as exc:
            print(f"  {int(fila['trial']):>5}  {familia:9s} {fila['feature_set']:16s}  FALLO: {type(exc).__name__}")
            continue

        marca = ""
        if champion:
            diff = oficial - champion[1]
            marca = f"  {diff:+.2f}" + ("  <-- SUPERA" if diff > 0 else "")
        print(f"  {int(fila['trial']):>5}  {familia:9s} {fila['feature_set']:16s}  "
              f"{fila['accuracy']:>6.2f}  {oficial:>6.2f}{marca}")
        filas.append({"trial": int(fila["trial"]), "familia": familia,
                      "feature_set": fila["feature_set"], "accuracy_barrido": fila["accuracy"],
                      "accuracy_oficial": oficial})

    if not filas:
        print("\nNinguno se pudo revalidar.")
        return

    resultado = pd.DataFrame(filas).sort_values("accuracy_oficial", ascending=False)
    salida = ROOT / "eda" / "reports" / "revalidacion.csv"
    resultado.to_csv(salida, index=False)
    print(f"\nGuardado en {salida}")

    # Cuanto se desvia la exploracion barata de la medicion oficial: dice si
    # el barrido sirve para preseleccionar o solo genera ruido.
    brecha = (resultado["accuracy_barrido"] - resultado["accuracy_oficial"]).abs().mean()
    print(f"Diferencia media entre exploracion y medicion oficial: {brecha:.2f} puntos")

    if champion:
        superan = resultado[resultado["accuracy_oficial"] > champion[1]]
        if superan.empty:
            print(f"\nNinguno supera al champion ({champion[1]:.2f}).")
            print("Eso tambien es un resultado: el champion actual no fue suerte.")
        else:
            mejor = superan.iloc[0]
            print(f"\n{len(superan)} candidato(s) superan al champion. El mejor:")
            print(f"  trial {mejor['trial']} ({mejor['familia']}, {mejor['feature_set']}): "
                  f"{mejor['accuracy_oficial']:.2f} vs {champion[1]:.2f}")
            print("\nPara promoverlo hay que agregarlo a build_candidates() en train.py")
            print("y correr train.py: la regla de promocion decide, no este script.")


if __name__ == "__main__":
    main()

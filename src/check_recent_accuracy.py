"""Disparador simple para el workflow de reentreno automatico.

El profesor confirmo por fuera del reto que metio el drift "al maximo" -un
cambio de FORMA en la demanda, no solo de nivel (ver docs/HALLAZGOS.md #28)-
y pidio explicitamente: reentrenar cada vez que el accuracy caiga por debajo
de un umbral fijo (80-85%), para mantenerlo alto mientras dure la fase mas
dura del drift.

Esto es deliberadamente MAS SIMPLE que `evaluate.py` (que usa un umbral
estadistico en desviaciones sobre una ventana de 24 ciclos, pensado para
degradacion lenta y sostenida). Aqui el pedido es distinto: reaccionar rapido
a una caida fuerte y reciente, con el mismo criterio de ventana que ya usa el
leaderboard (ultimos 6 ciclos oficiales resueltos). No reemplaza a
evaluate.py, corre aparte y con un proposito mas directo.

train.py ya trae su propia red de seguridad (una version nueva solo reemplaza
al champion si lo SUPERA en la misma validacion cruzada), asi que disparar el
reentreno de mas nunca puede degradar lo que ya funciona: en el peor caso se
guarda como "candidate" sin tocar nada.

Imprime el accuracy promedio de la ventana y termina con:
  exit 0  -> por encima del umbral, no hace falta reentrenar.
  exit 1  -> por debajo del umbral (o sin datos suficientes para decidir con
             confianza), hay que reentrenar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# El profesor dio un rango (80-85%); se toma el punto medio conservador para
# no disparar reentrenos de mas por ruido de un solo ciclo, pero reaccionar
# antes de perforar el piso de abajo.
UMBRAL_ACCURACY = float(os.environ.get("UMBRAL_REENTRENO", "82.5"))

# Mismo tamano de ventana que usa el leaderboard oficial ("ultimos 6 ciclos"),
# para que "cayo por debajo del umbral" signifique lo mismo aqui y alla.
CICLOS_VENTANA = 6


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def accuracy_ciclos_recientes(client: httpx.Client, url: str, n: int) -> pd.DataFrame:
    """Accuracy general (station_id nulo, el promedio ya calculado) de los
    `n` ciclos OFICIALES mas recientes con metrica calculada.

    Un mismo ciclo puede tener mas de una fila (evaluate.py recalcula al
    llegar mas verdad): se pide el doble de filas y se deduplica por
    cycle_id quedandose con la mas reciente, para que "ultimos N ciclos"
    signifique N ciclos distintos, no N filas."""
    r = client.get(
        f"{url}/rest/v1/cycle_metrics",
        headers=supabase_headers(),
        params={
            "select": "cycle_id,accuracy,computed_at",
            "station_id": "is.null",
            "cycle_id": "like.cyc_official-*",
            "order": "computed_at.desc",
            "limit": str(n * 3),
        },
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df = df.sort_values("computed_at", ascending=False).drop_duplicates(subset="cycle_id", keep="first")
    return df.head(n)


def main() -> None:
    url = os.environ["SUPABASE_URL"]
    with httpx.Client(timeout=30.0) as client:
        df = accuracy_ciclos_recientes(client, url, CICLOS_VENTANA)

    if df.empty or df["accuracy"].isna().all():
        print(
            f"Sin ciclos oficiales resueltos todavia (0 filas). No se puede decidir con "
            f"confianza: se dispara el reentreno por precaucion (mejor sobrar que faltar)."
        )
        sys.exit(1)

    df = df.dropna(subset=["accuracy"])
    media = float(df["accuracy"].mean())
    peor = float(df["accuracy"].min())
    n = len(df)

    print(f"Ultimos {n} ciclo(s) oficiales resueltos:")
    for _, fila in df.iterrows():
        print(f"  {fila['cycle_id'][-11:-1]}: {fila['accuracy']:.2f}")
    print(f"\nPromedio: {media:.2f}  |  peor ciclo: {peor:.2f}  |  umbral: {UMBRAL_ACCURACY:.1f}")

    if media < UMBRAL_ACCURACY:
        print(f"\nPor DEBAJO del umbral ({media:.2f} < {UMBRAL_ACCURACY:.1f}): hay que reentrenar.")
        sys.exit(1)

    print(f"\nPor ENCIMA del umbral ({media:.2f} >= {UMBRAL_ACCURACY:.1f}): no hace falta reentrenar.")
    sys.exit(0)


if __name__ == "__main__":
    main()

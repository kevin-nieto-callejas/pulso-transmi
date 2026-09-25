"""Simulacro de un ciclo completo de competencia, de punta a punta.

Por que existe: `evaluate.py` esta probado con 11 tests, pero nunca ha escrito
una fila real en `prediction_evaluations` ni en `cycle_metrics`. Los tests
usan datos inventados en memoria; no tocan el esquema, ni los tipos de
Postgres, ni las llaves foraneas. Exactamente ese hueco fue el que dejo pasar
el bug de paginacion: funcionaba con conjuntos pequenos y se rompia con
volumen real.

Ademas, la ronda de practica pidio 12 predicciones todas a +15 min. La
competencia real pide 48 con cuatro horizontes distintos, y ese camino nunca
se ha ejercitado.

Como funciona: se ancla el ciclo en un instante PASADO del historico, de modo
que los valores "reales" de los targets no hay que inventarlos -ya existen en
`observations`-. Asi el emparejamiento prediccion/realidad es genuino y las
metricas que salen son comprobables a mano.

Al terminar borra todo lo que creo. Dejar datos simulados en la base seria
ensuciar la evidencia del proyecto y, peor, inflar el dashboard con numeros
que no son de la competencia.

Uso:
    python src/simulate_cycle.py            # simula y limpia
    python src/simulate_cycle.py --conservar  # deja los datos para ver el dashboard
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import emparejar_con_la_realidad, metricas_por_ciclo, supabase_headers, traer  # noqa: E402
from features import HORIZONS_MINUTES, wape_accuracy  # noqa: E402
from infer import build_anchor_features, build_batch_predictions  # noqa: E402
from perfil import PerfilAdaptativo  # noqa: E402
from predict import get_champion, load_model_from_storage  # noqa: E402


def escribir(client: httpx.Client, url: str, tabla: str, filas: list[dict]) -> None:
    r = client.post(f"{url}/rest/v1/{tabla}", headers={**supabase_headers(), "Prefer": "return=minimal"}, json=filas)
    if r.status_code >= 300:
        raise RuntimeError(f"No se pudo escribir en {tabla}: {r.status_code} {r.text[:400]}")


def borrar(client: httpx.Client, url: str, tabla: str, filtro: dict) -> int:
    r = client.delete(f"{url}/rest/v1/{tabla}", headers={**supabase_headers(), "Prefer": "return=representation"},
                      params=filtro)
    if r.status_code >= 300:
        raise RuntimeError(f"No se pudo limpiar {tabla}: {r.status_code} {r.text[:300]}")
    return len(r.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulacro de ciclo completo")
    parser.add_argument("--conservar", action="store_true", help="no borrar los datos simulados al terminar")
    parser.add_argument("--horas-atras", type=float, default=1.0,
                        help="cuantas horas antes del ultimo dato anclar el ciclo (cambia la hora del dia simulada)")
    args = parser.parse_args()

    url = os.environ["SUPABASE_URL"].rstrip("/")
    cycle_id = f"cyc_sim_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"=== SIMULACRO {cycle_id} ===\n")

    with httpx.Client(timeout=120.0) as client:
        # 1. Anclar en el pasado: asi los valores reales de los 4 horizontes ya
        #    existen en la base y no hay que inventar nada.
        obs = traer(client, url, "observations", {"select": "station_id,observed_at,demand"})
        obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
        data_cutoff = obs["observed_at"].max() - pd.Timedelta(hours=args.horas_atras)
        estaciones = sorted(obs["station_id"].unique())
        print(f"1. Ancla: {data_cutoff} ({len(estaciones)} estaciones)")

        # 2. Ciclo con 48 targets: 12 estaciones x 4 horizontes. La practica
        #    solo probo 12 targets a un mismo horizonte.
        targets = [
            {"station_id": s, "target_at": (data_cutoff + pd.Timedelta(minutes=h)).isoformat()}
            for s in estaciones for h in HORIZONS_MINUTES
        ]
        print(f"2. Targets generados: {len(targets)} (horizontes {list(HORIZONS_MINUTES)})")
        assert len(targets) == 48, f"Se esperaban 48 targets, hay {len(targets)}"

        escribir(client, url, "cycles", [{
            "cycle_id": cycle_id, "opens_at": data_cutoff.isoformat(), "data_cutoff": data_cutoff.isoformat(),
            "closes_at": (datetime.now(timezone.utc) + timedelta(minutes=25)).isoformat(), "status": "closed",
        }])

        # 3. Predecir con el champion real, por el mismo camino que infer.py
        champion = get_champion()
        bundle = load_model_from_storage(champion["artifact_location"])
        anclas, observaciones = build_anchor_features(client, url, data_cutoff)
        # El simulacro tiene que recorrer el MISMO camino que la entrega real,
        # perfil incluido: si aqui se predijera solo con el champion, dejaria
        # de detectar los problemas que importan.
        predicciones = build_batch_predictions(
            bundle["model"], bundle["feature_columns"], anclas, targets, data_cutoff,
            PerfilAdaptativo(observaciones),
        )
        horizontes_usados = sorted({
            round((pd.Timestamp(p["target_at"]) - data_cutoff).total_seconds() / 60) for p in predicciones
        })
        print(f"3. Predicciones: {len(predicciones)} con horizontes {horizontes_usados}")
        assert horizontes_usados == list(HORIZONS_MINUTES), "Los 4 horizontes no llegaron completos"

        escribir(client, url, "predictions", [{
            "cycle_id": cycle_id, "station_id": p["station_id"], "target_at": p["target_at"],
            "horizon_minutes": round((pd.Timestamp(p["target_at"]) - data_cutoff).total_seconds() / 60),
            "predicted_value": p["value"], "model_version_id": champion["version_id"],
        } for p in predicciones])

        # 4. Evaluar contra la realidad (que ya estaba en la base)
        guardadas = traer(client, url, "predictions",
                          {"select": "id,cycle_id,station_id,target_at,predicted_value",
                           "cycle_id": f"eq.{cycle_id}"})
        evaluadas = emparejar_con_la_realidad(guardadas, traer(client, url, "observations",
                                                               {"select": "station_id,observed_at,demand"}))
        print(f"4. Emparejadas con realidad: {len(evaluadas)} de {len(guardadas)}")
        if evaluadas.empty:
            raise RuntimeError("Ninguna prediccion encontro su valor real: el emparejamiento esta roto")

        escribir(client, url, "prediction_evaluations", [{
            "prediction_id": int(r.id), "actual_value": float(r.actual_value),
            "absolute_error": float(r.absolute_error),
        } for r in evaluadas.itertuples()])

        metricas = metricas_por_ciclo(evaluadas)
        escribir(client, url, "cycle_metrics", [{
            "cycle_id": r["cycle_id"], "station_id": r["station_id"],
            "wape": None if pd.isna(r["wape"]) else float(r["wape"]), "accuracy": float(r["accuracy"]),
        } for _, r in metricas.iterrows()])

        # 5. Comprobar el resultado a mano, sin usar el mismo codigo
        total = metricas[metricas["station_id"].isna()].iloc[0]["accuracy"]
        a_mano = sum(
            wape_accuracy(g["actual_value"].values, g["predicted_value"].values)
            for _, g in evaluadas.groupby("station_id")
        ) / evaluadas["station_id"].nunique()
        print(f"\n5. Accuracy del ciclo simulado: {total:.2f}")
        print(f"   Recalculada aparte        : {a_mano:.2f}")
        assert abs(total - a_mano) < 0.01, "La metrica guardada no coincide con el calculo directo"

        por_horizonte = evaluadas.assign(
            h=(pd.to_datetime(evaluadas["target_at"], utc=True) - data_cutoff).dt.total_seconds() / 60
        ).groupby("h").apply(
            lambda g: wape_accuracy(g["actual_value"].values, g["predicted_value"].values), include_groups=False
        )
        print("\n   Por horizonte:")
        for h, acc in por_horizonte.items():
            print(f"     +{int(h):>2} min  {acc:.2f}")

        peores = metricas[metricas["station_id"].notna()].nsmallest(3, "accuracy")
        print("\n   Estaciones mas dificiles:")
        for _, r in peores.iterrows():
            print(f"     {r['station_id']}  {r['accuracy']:.2f}")

        # 6. Limpiar
        if args.conservar:
            print(f"\n6. Datos CONSERVADOS ({cycle_id}). Para borrarlos despues:")
            print(f"   python src/simulate_cycle.py --limpiar {cycle_id}")
            return

        print("\n6. Limpiando el simulacro...")
        ids = [int(i) for i in guardadas["id"]]
        for pid in ids:
            borrar(client, url, "prediction_evaluations", {"prediction_id": f"eq.{pid}"})
        n_m = borrar(client, url, "cycle_metrics", {"cycle_id": f"eq.{cycle_id}"})
        n_p = borrar(client, url, "predictions", {"cycle_id": f"eq.{cycle_id}"})
        n_c = borrar(client, url, "cycles", {"cycle_id": f"eq.{cycle_id}"})
        print(f"   Borradas: {len(ids)} evaluaciones, {n_m} metricas, {n_p} predicciones, {n_c} ciclo")

        restantes = traer(client, url, "predictions", {"select": "id", "cycle_id": f"eq.{cycle_id}"})
        assert restantes.empty, "Quedaron datos del simulacro en la base"
        print("   Base limpia: no quedo rastro del simulacro.")

    print("\n=== SIMULACRO EXITOSO ===")
    print("El camino completo -ciclo, 48 targets, 4 horizontes, prediccion,")
    print("emparejamiento con la realidad, metricas y limpieza- funciona.")


if __name__ == "__main__":
    main()

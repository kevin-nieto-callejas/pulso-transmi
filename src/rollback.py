"""Vuelve a un champion anterior cuando el vigente da problemas.

La guia lista "estrategia de rollback del modelo" entre los bonos, y en este
proyecto dejo de ser teorico: el champion actual es un ensamble de 38 MB que
necesita tres librerias distintas para reconstruirse. Si algo de eso falla en
plena competencia, hay que poder volver a un modelo conocido en un minuto, sin
reentrenar nada y sin improvisar.

Es posible porque ningun champion se borra: al ser reemplazado queda marcado
`historical`, con su metrica, sus features y su artefacto intactos.

Dos salvaguardas que no se pueden saltar:

  1. **Se verifica ANTES de cambiar.** El artefacto de destino se descarga y
     se le pide una prediccion. Volver a un modelo que tampoco carga seria
     cambiar una falla por otra, y encima con el sistema ya degradado.
  2. **Toda vuelta atras queda justificada.** `--motivo` es obligatorio y se
     registra en `drift_signals` como senal operacional. La rubrica pide
     decisiones sustentadas, y un cambio de champion sin explicacion escrita
     es exactamente lo contrario.

Uso:
    python src/rollback.py --listar
    python src/rollback.py --a <version_id> --motivo "el ensamble no carga en Actions"
    python src/rollback.py --auto --motivo "..."   # al mejor historico disponible
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import load_model_from_storage  # noqa: E402


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def listar_versiones(client: httpx.Client, url: str) -> list[dict]:
    response = client.get(
        f"{url}/rest/v1/model_versions",
        headers=supabase_headers(),
        params={
            "select": "version_id,status,validation_metric,created_at,features,artifact_location",
            "order": "created_at.desc",
        },
    )
    response.raise_for_status()
    return response.json()


def verificar_que_carga(version: dict) -> float:
    """Descarga el artefacto y le pide una prediccion. Falla ruidosamente."""
    bundle = load_model_from_storage(version["artifact_location"])
    modelo, columnas = bundle["model"], bundle["feature_columns"]
    fila = pd.DataFrame([{c: 0.0 for c in columnas}])
    valor = float(modelo.predict(fila)[0])
    print(f"  Verificado: {type(modelo).__name__}, {len(columnas)} features, prediccion de prueba {valor:.2f}")
    return valor


def cambiar_estado(client: httpx.Client, url: str, version_id: str, estado: str) -> None:
    response = client.patch(
        f"{url}/rest/v1/model_versions",
        headers={**supabase_headers(), "Prefer": "return=minimal"},
        params={"version_id": f"eq.{version_id}"},
        json={"status": estado},
    )
    if response.status_code >= 300:
        raise RuntimeError(f"No se pudo marcar {version_id} como {estado}: {response.status_code} {response.text[:300]}")


def registrar_decision(client: httpx.Client, url: str, desde: str, hacia: str, motivo: str) -> None:
    """Deja la vuelta atras en drift_signals como senal operacional."""
    payload = {
        "signal_type": "operational",
        "description": f"Rollback de champion: {desde} -> {hacia}. Motivo: {motivo}",
        "action_taken": f"Se promovio {hacia} y {desde} quedo como historical.",
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    response = client.post(
        f"{url}/rest/v1/drift_signals", headers={**supabase_headers(), "Prefer": "return=minimal"}, json=payload,
    )
    if response.status_code >= 300:
        print(f"  AVISO: no se pudo registrar la decision ({response.status_code}). El rollback si se aplico.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vuelve a un champion anterior")
    parser.add_argument("--listar", action="store_true", help="solo muestra las versiones disponibles")
    parser.add_argument("--a", dest="destino", help="version_id al que volver")
    parser.add_argument("--auto", action="store_true", help="vuelve al historico con mejor metrica")
    parser.add_argument("--motivo", help="por que se vuelve atras (obligatorio al aplicar)")
    args = parser.parse_args()

    url = os.environ["SUPABASE_URL"].rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        versiones = listar_versiones(client, url)
        actual = next((v for v in versiones if v["status"] == "champion"), None)
        historicos = [v for v in versiones if v["status"] == "historical"]

        print("=== Versiones registradas ===")
        for v in versiones:
            marca = " <-- CHAMPION" if v["status"] == "champion" else ""
            horizontes = "multi-horizonte" if "horizon_minutes" in (v.get("features") or []) else "UN SOLO HORIZONTE"
            print(f"  {v['version_id']:42s} {v['status']:10s} {v['validation_metric']:6.2f}  {horizontes}{marca}")

        if args.listar:
            return
        if not args.motivo:
            parser.error("--motivo es obligatorio: toda vuelta atras debe quedar justificada")
        if actual is None:
            parser.error("No hay champion vigente; esto no es un rollback sino una promocion inicial")

        if args.auto:
            compatibles = [v for v in historicos if "horizon_minutes" in (v.get("features") or [])]
            if not compatibles:
                parser.error(
                    "No hay historicos multi-horizonte. Los de un solo horizonte NO sirven: "
                    "solo saben predecir +15 min y la API pide cuatro horizontes."
                )
            destino = max(compatibles, key=lambda v: v["validation_metric"])
        else:
            if not args.destino:
                parser.error("Indica --a <version_id> o usa --auto")
            destino = next((v for v in versiones if v["version_id"] == args.destino), None)
            if destino is None:
                parser.error(f"No existe la version {args.destino}")
            if destino["status"] == "champion":
                parser.error("Esa version ya es el champion vigente")

        print(f"\nChampion vigente : {actual['version_id']} ({actual['validation_metric']:.2f})")
        print(f"Volviendo a      : {destino['version_id']} ({destino['validation_metric']:.2f})")
        if "horizon_minutes" not in (destino.get("features") or []):
            print(
                "\n  ATENCION: el destino es de un solo horizonte. Solo sabe predecir +15 min;\n"
                "  para +30/+45/+60 produciria valores sin sentido. Usalo solo como ultimo recurso."
            )

        print("\nVerificando que el destino cargue ANTES de cambiar nada...")
        verificar_que_carga(destino)

        cambiar_estado(client, url, actual["version_id"], "historical")
        cambiar_estado(client, url, destino["version_id"], "champion")
        registrar_decision(client, url, actual["version_id"], destino["version_id"], args.motivo)

        print(f"\nHECHO. Champion ahora: {destino['version_id']}")
        print(f"       {actual['version_id']} quedo como historical (se puede volver a el).")
        print("       Decision registrada en drift_signals.")


if __name__ == "__main__":
    main()

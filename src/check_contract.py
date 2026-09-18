"""Vigila cambios del profesor que puedan rompernos.

El docente versiona su plataforma con frecuencia (portal, contrato, frontend).
Si cambia el contrato de la API, nuestro pipeline puede dejar de funcionar
justo durante la competencia, y lo peor es que algunos cambios NO producen un
error evidente: simplemente dejariamos de entregar, o entregariamos mal.

Este script compara la realidad contra `contract/expected.json` (lo ultimo que
revisamos y sabemos que funciona) y falla con un mensaje explicito cuando algo
cambio. Corre en GitHub Actions varias veces al dia: al fallar, GitHub envia
correo automaticamente.

No pretende adivinar si el cambio es bueno o malo - solo obliga a que alguien
lo mire antes de que nos sorprenda en un ciclo real. Cuando revisemos y
adaptemos el codigo, se actualiza `contract/expected.json` a proposito.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
GITHUB_API = "https://api.github.com"


def cargar_esperado() -> dict:
    return json.loads((ROOT / "contract" / "expected.json").read_text(encoding="utf-8"))


def revisar_api(client: httpx.Client, esperado: dict) -> list[str]:
    hallazgos = []
    spec = client.get(f"{API_URL}/openapi.json").json()

    version = spec.get("info", {}).get("version")
    if version != esperado["api_version"]:
        hallazgos.append(
            f"La version del API cambio: {esperado['api_version']} -> {version}. "
            "Revisar el changelog del profesor antes de confiar en el pipeline."
        )

    rutas = set(spec.get("paths", {}))
    for endpoint in esperado["endpoints_requeridos"]:
        if endpoint not in rutas:
            hallazgos.append(f"DESAPARECIO el endpoint que usamos: {endpoint}")

    schemas = spec.get("components", {}).get("schemas", {})
    for nombre_schema, campos_esperados, etiqueta in (
        ("SubmissionInput", esperado["campos_submission"], "submission"),
        ("PredictionInput", esperado["campos_prediction"], "prediccion"),
    ):
        propiedades = set(schemas.get(nombre_schema, {}).get("properties", {}))
        if not propiedades:
            hallazgos.append(f"Ya no existe el schema {nombre_schema} en el contrato.")
            continue
        faltantes = [c for c in campos_esperados if c not in propiedades]
        if faltantes:
            hallazgos.append(f"Campos de {etiqueta} que ya no existen: {faltantes}")
        nuevos_requeridos = [
            c for c in schemas.get(nombre_schema, {}).get("required", []) if c not in campos_esperados
        ]
        if nuevos_requeridos:
            hallazgos.append(
                f"El contrato ahora EXIGE campos nuevos en {etiqueta} que no enviamos: {nuevos_requeridos}"
            )
    return hallazgos


def revisar_ciclo_abierto(client: httpx.Client, esperado: dict) -> list[str]:
    """Si hay un ciclo abierto, confirma que trae los campos que leemos."""
    response = client.get(f"{API_URL}/v1/forecast-cycles/current")
    if response.status_code == 404:
        return []  # sin ciclo abierto no hay nada que validar; no es un problema
    response.raise_for_status()
    ciclo = response.json()

    faltantes = [c for c in esperado["campos_ciclo_que_usamos"] if c not in ciclo]
    if faltantes:
        return [f"El ciclo abierto ya no trae campos que usamos: {faltantes}. Payload: {list(ciclo)}"]

    targets = ciclo.get("targets") or []
    if targets and not {"station_id", "target_at"} <= set(targets[0]):
        return [f"Los targets cambiaron de forma. Ahora traen: {list(targets[0])}"]
    return []


def revisar_repo_del_profesor(client: httpx.Client, esperado: dict) -> list[str]:
    """Avisa si hay commits nuevos desde el ultimo que revisamos."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    repo = esperado["upstream_repo"]
    response = client.get(f"{GITHUB_API}/repos/{repo}/commits", headers=headers, params={"per_page": 20})
    if response.status_code >= 300:
        print(f"  (no se pudo consultar {repo}: {response.status_code}; se omite esta revision)")
        return []

    commits = response.json()
    conocido = esperado["upstream_last_reviewed_sha"]
    nuevos = []
    for commit in commits:
        if commit["sha"] == conocido:
            break
        nuevos.append(f"    - {commit['commit']['author']['date']}  {commit['commit']['message'].splitlines()[0]}")

    if nuevos:
        return [
            f"El profesor publico {len(nuevos)} commit(s) nuevos en {repo} desde el ultimo que revisamos:\n"
            + "\n".join(nuevos)
            + "\n    Revisar si afectan el contrato y luego actualizar contract/expected.json."
        ]
    return []


def main() -> None:
    esperado = cargar_esperado()
    print(f"Contrato de referencia: API {esperado['api_version']}, upstream {esperado['upstream_last_reviewed_sha'][:8]}")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        hallazgos = revisar_api(client, esperado)
        hallazgos += revisar_ciclo_abierto(client, esperado)
        hallazgos += revisar_repo_del_profesor(client, esperado)

    if not hallazgos:
        print("Sin novedades: el contrato y el repo del profesor siguen como los revisamos.")
        return

    print("\n=== CAMBIOS DETECTADOS DEL LADO DEL PROFESOR ===")
    for hallazgo in hallazgos:
        print(f"  * {hallazgo}")
    print(
        "\nEsto NO significa que algo este roto, significa que hay que mirarlo.\n"
        "Cuando se revise y (si hace falta) se adapte el codigo, actualizar\n"
        "contract/expected.json para volver a poner esta alarma en verde."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

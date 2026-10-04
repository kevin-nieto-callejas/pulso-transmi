# Traspaso: Pulso TransMi (Kevin Nieto) — 2026-10-04 ~02:25 UTC

Pega este archivo al iniciar un chat nuevo de Claude Code, con el directorio de trabajo
`\\wsl.localhost\Ubuntu\home\asus\code\clase3107\pulso-transmi-sdk`.

## Qué es
Competencia MLOps del profe (Universidad Externado): predecir demanda de 12 estaciones de
TransMilenio, 4 horizontes (+15/+30/+45/+60 min) = 48 predicciones por ciclo. Un ciclo por
hora real, ventana de entrega de 25 min. Métrica: accuracy = 100·(1−WAPE), acumulada.

**Cierre final: domingo 4 oct 2026, 23:59 Bogotá = lunes 5 oct 04:59 UTC.**

## Dónde está todo
| Qué | Dónde |
|---|---|
| Repo (público) | https://github.com/kevin-nieto-callejas/pulso-transmi (rama `main`) |
| Copia local | `\\wsl.localhost\Ubuntu\home\asus\code\clase3107\pulso-transmi-sdk` |
| Python venv | `.venv\Scripts\python.exe` dentro del repo (venv Windows) |
| Credenciales | `.env` en la raíz del repo (API Pulso, Supabase). En GitHub: Secrets |
| API del profe | https://pulso-transmi.72-60-245-2.sslip.io (portal, `/openapi.json`, `/v1/leaderboard`) |
| Repo del profe | https://github.com/uexternadojz/pulso-transmi — leer `docs/fase-final.md` |
| Supabase | proyecto `bawwhejgcvlawfqualrj` (tablas: observations, predictions, submissions, cycle_metrics, prediction_evaluations, collector_runs, model_versions) |
| Dashboard propio | https://pulso-transmi-one.vercel.app |
| Docs del proyecto | `docs/HALLAZGOS.md` (hallazgos #1–#36), `docs/BITACORA.md`, `docs/INFORME_FINAL.md`, `docs/RUNBOOK.md`, `docs/HANDOFF.md` |
| Skill de envío | `.claude/skills/enviar-predicciones/SKILL.md` |

## Cómo se entrega (automático, en la nube)
- **`.github/workflows/relay.yml` (NUEVO, el principal):** un job que vive ~5h40m, sondea
  `/v1/forecast-cycles/current` cada 45 s; si hay ciclo nuevo corre `src/ingest.py` y luego
  `src/infer.py`. Al terminar se re-dispara solo (`gh workflow run relay.yml`). El cron
  `7,27,47 * * * *` solo lo revive si la cadena se rompe. **No depende del PC.**
- `inference.yml`: el flujo viejo (cron cada 5 min, pero GitHub solo lo dispara cada 3–5 h).
  Sirve de respaldo y para disparar a mano: `gh workflow run inference.yml -R kevin-nieto-callejas/pulso-transmi`.
- Antes, las entregas puntuales venían de un vigilante en el PC (workflow_dispatch). Al apagarlo
  se perdieron los ciclos virtuales 12 y 14 → por eso existe relay.yml.
- Idempotencia: la llave es (cycle_id, version del champion); `infer.py` no reenvía si ya está en `submissions`.

## Modelo (cadena en `src/infer.py::_capa_adaptativa`)
1. **Onda corta** (`perfil.prediccion_periodica`, hallazgo #35): copia el valor de hace P pasos
   (P≈16 = 4 h) si acertó ≥80% en las últimas 12 h **y ahora también en las últimas 2 h**.
2. **Persistencia + tendencia** (`perfil.persistencia_tendencia`, NUEVO): último valor real +
   0.5 × pendiente de la última hora × pasos al target.
3. Extrapolación extrema / mezcla champion (CatBoost `catboost_sin_semanal-20260921T161638Z`) + perfil.

## Lo que pasó hoy (4 oct) y los arreglos (commits en main)
- `dad2211` relay.yml (entregas sin PC).
- `2dcd8aa` **Contrato de observación v2**: desde 2026-09-20T12:00Z virtual el stream trae
  `measurement.value` (texto, o null con `quality=missing`) en vez de `demand`. `ingest.py`
  se caía con KeyError → se predecía con datos de 3 h atrás. Ahora lee v1 y v2 y omite faltantes.
- `18d5f68` **Revisión 4 del drift** (activada 3 oct ~22:50 UTC): se acabó la onda de 4 h; ahora
  hay tendencias lentas de varias horas. La onda corta seguía activa y acertaba ~0%.
  Backtest con datos rev 4 (cortes 12:00–14:00 virtual): onda vieja ≈0–33; mezcla champion+perfil
  64; persistencia 72.1; persistencia+½tendencia 72.6. Pipeline nuevo desde 13:15: 78–83.

## Cómo vamos (leaderboard oficial, 4 oct 01:58 UTC)
Puesto **6 de 32**, accuracy acumulada **76.26**, cobertura 98.9% (186/188 ciclos).
1º John Bernal 81.02 · 2º Isaias Céspedes 80.44 · 3º Allison Loango 78.64 · 4º Daniela González 78.59 ·
5º Lis Sánchez 76.57 · **6º Kevin 76.26** · 7º Mateo Hoyos 76.21.
Historia: ~85 en régimen normal; el 1 oct (drift rev 3, antes de la onda corta) se promedió 38
y eso hundió el acumulado; con la onda corta, 37 ciclos seguidos en 90–94.
Quedan ~27 ciclos: remontar al 1º es muy difícil, pero el 5º está a 0.31 y el 7º a 0.05.

## Pendientes / ideas para seguir
1. **Verificar el primer ciclo con el código nuevo** (abre ~02:49 UTC 4 oct): en el log del relay
   deben salir líneas `TENDENCIA ->` y `Submission ACEPTADA`. Comprobar en Supabase:
   `select submitted_at, cycle_id from submissions order by submitted_at desc limit 5;`
2. Con más datos rev 4, re-medir candidatos (script de referencia: probar persistencia, fracción de
   tendencia 0.3–0.7, ventana de pendiente, detectar si aparece una nueva periodicidad de 6–12 h —
   el detector actual solo busca P entre 2 h y 6 h).
3. Observaciones se liberan cada 30 min: el último dato puede ir 15–30 min atrás del data_cutoff.
4. ~~Documentar para la nota~~ **Hecho 4-oct ~02:40 UTC:** hallazgos #37 (v2 + alarma en
   rojo 5 días), #38 (rev 4), #39 (relay sin PC); sección "Fase final" en INFORME_FINAL;
   bitácora, README, RUNBOOK y HANDOFF al día; `contract/expected.json` en 0.9.0 (alarma en verde).
5. ~~Cambios locales sin commitear~~ **Hecho:** revisados e integrados en el mismo commit.
   Además `relay.yml` ahora corre `evaluate.py` tras cada entrega.
6. Seguridad: revocar el PAT de GitHub que se pegó en un chat anterior
   y considerar rotar la service-role key de Supabase y la API key de Pulso al terminar.

## Comandos útiles
```powershell
gh run list -R kevin-nieto-callejas/pulso-transmi --workflow=relay.yml --limit 5
gh run list -R kevin-nieto-callejas/pulso-transmi --workflow=inference.yml --limit 5
Invoke-RestMethod https://pulso-transmi.72-60-245-2.sslip.io/v1/clock
# correr local (carga .env):
Get-Content .env | % { if ($_ -match '^\s*([A-Z_]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }
.\.venv\Scripts\python.exe src\ingest.py ; .\.venv\Scripts\python.exe src\infer.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

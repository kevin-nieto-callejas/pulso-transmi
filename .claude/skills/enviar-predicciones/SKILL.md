---
name: enviar-predicciones
description: Enviar predicciones a la API de Pulso TransMi. Úsala cuando haya que entregar un ciclo (manual o automático), diagnosticar por qué una submission fue rechazada, o verificar que una entrega quedó aceptada. Cubre el contrato exacto, las reglas que no se pueden romper y los errores reales que ya nos pasaron.
---

# Enviar predicciones a Pulso TransMi

## Lo esencial en tres frases

1. La API dice qué predecir; nosotros **nunca** lo inventamos.
2. Un ciclo se entrega completo o no se entrega: 48 predicciones (12 estaciones × 4 horizontes), o las que el ciclo pida.
3. Reintentar con la misma llave de idempotencia es seguro; reintentar con contenido distinto es un conflicto.

## Cómo se envía

Todo el flujo está en [`src/infer.py`](../../../src/infer.py):

```bash
python src/infer.py
```

Requiere en el entorno:

| Variable | Para qué |
|---|---|
| `PULSO_API_KEY` | Identidad del participante (`ptm_live_...`) |
| `SUPABASE_URL` | Leer el histórico y guardar la evidencia |
| `SUPABASE_SERVICE_ROLE_KEY` | Escribir en Supabase |
| `PULSO_POLL_MINUTES` | Opcional: minutos que la corrida se queda vigilando |

En automático corre vía `.github/workflows/inference.yml` (cron en los minutos
`3,13,23,33,43,53`), con las credenciales en GitHub Secrets.

## Qué hace, paso a paso

1. **Consulta el reloj y el ciclo vigente** (`/v1/clock`, `/v1/forecast-cycles/current`).
   Si no hay ciclo abierto, **termina sin error** — es el estado normal.
2. **Verifica si ese ciclo ya se entregó** con el champion actual (consulta la
   tabla `submissions`). Si ya está, no reenvía.
3. **Carga el champion** desde Supabase Storage (con caché local).
4. **Reconstruye las features** tal como se veían en `data_cutoff`, usando
   hasta 8 días de histórico (hacen falta para `lag_672` y `lag_1344`).
5. **Arma el batch** con los targets exactos que pidió la API.
6. **Envía** a `POST /v1/submissions` con la cabecera `Idempotency-Key`.
7. **Guarda la evidencia**: ciclo, predicciones y recibo en Supabase.

## El contrato

`POST /v1/submissions`

Cabeceras: `Authorization: Bearer <PULSO_API_KEY>` y `Idempotency-Key` (8-128 caracteres).

```json
{
  "schema_version": "1.0",
  "cycle_id": "cyc_...",
  "client_run_id": "<uuid único por intento>",
  "data_cutoff": "<el que devolvió la API>",
  "model": {
    "version": "<version_id del champion>",
    "trained_at": "...",
    "training_data_end": "...",
    "git_commit": "<hash corto>"
  },
  "predictions": [
    {"station_id": "03000", "target_at": "...", "value": 182.32}
  ]
}
```

`value` debe ser finito y estar entre 0 y 100000.

## Reglas que no se pueden romper

- **Nunca fabricar `cycle_id`, `data_cutoff` ni los targets** a partir de la
  hora local. El cron de GitHub se retrasa; la API es la autoridad.
- **Solo información disponible hasta `data_cutoff`.** Usar datos posteriores
  es hacer trampa y además no existirían en la competencia real.
- **No agregar ni quitar targets.** La API acepta el batch completo o lo
  rechaza completo.
- **Máximo 3 intentos válidos por ciclo.** El último aceptado antes del cierre
  es el oficial.
- **La misma llave de idempotencia para el mismo contenido.** Aquí se deriva
  de `(cycle_id, version_id del champion)`, así que un reintento devuelve el
  mismo recibo sin gastar un intento.

## Verificar que quedó entregado

```bash
# Recibo oficial
curl -s "https://pulso-transmi.72-60-245-2.sslip.io/v1/submissions/<submission_id>" \
  -H "Authorization: Bearer $PULSO_API_KEY"
```

También en el portal (https://pulso-transmi.72-60-245-2.sslip.io/), recuadro
"Última entrega", y en la tabla `submissions` de Supabase.

Un recibo aceptado trae `"status": "accepted"` e `"is_official": true`.
**Aceptada no significa correcta**: solo confirma que llegó a tiempo, con
identidad válida y estructura completa.

## Errores y qué significan

| Error | Causa | Qué hacer |
|---|---|---|
| `401 invalid_api_key` | Falta, está mal escrita o fue revocada | Revisar `PULSO_API_KEY`; se puede rotar en el portal (`/v1/portal/api-key/rotate`) |
| `404 no_open_cycle` | No hay ciclo abierto | No es un error: terminar sin fallar |
| `409 attempt_limit_reached` | Ya se usaron los 3 intentos | No insistir; revisar por qué se gastaron |
| `409` por idempotencia | Misma llave con contenido distinto | No cambiar el contenido y reusar llave: o se reusa igual, o se cambia la llave |
| `422 invalid_target_set` | Targets faltantes, duplicados o con timestamps alterados | Tomar los targets tal cual los da la API |

## Problemas reales que ya nos pasaron

Están documentados en [`docs/BITACORA.md`](../../../docs/BITACORA.md). Los que
afectan directamente al envío:

- **Solo se cargaba 1 de las 12 estaciones.** PostgREST nunca devuelve más de
  1000 filas por respuesta, sin importar el `limit`. Hay que paginar con
  cabeceras `Range` (ver `fetch_all_rows`).
- **El horizonte se medía desde `data_cutoff`, no desde el ancla.** Si el
  collector va atrasado, la última observación queda antes del corte y el
  modelo recibiría "salta 15 min" cuando debe saltar 45. Se mide contra el
  timestamp real del ancla y se avisa si hay atraso.
- **GitHub descarta corridas programadas** bajo carga. Por eso el cron dispara
  cada 10 minutos y cada corrida vigila 8 minutos más.
- **El artefacto del champion pesa 38 MB.** Sin caché serían ~6 GB por semana
  de competencia, por encima del plan gratuito de Supabase.

## Antes de tocar el código de envío

Corre las pruebas: `pytest tests/test_infer.py`. Cubren el camino sin ciclo
abierto, el armado del batch, el recorte de valores, la llave de idempotencia,
la detección de "ya entregado" y la paginación.

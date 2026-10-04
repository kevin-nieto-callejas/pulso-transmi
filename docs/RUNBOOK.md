# Runbook — día de competencia

Qué mirar, qué es normal y qué hacer cuando algo falla. Escrito para leerse
con prisa, no de corrido.

**Regla general: el sistema entrega solo.** Si no hay nada roto, lo correcto
es no tocar nada. Las intervenciones manuales durante la competencia son la
forma más fácil de perder una ventana.

## Enlaces

| Qué | Dónde |
|---|---|
| Portal (entregas, API key) | https://pulso-transmi.72-60-245-2.sslip.io/ |
| Dashboard del sistema | https://pulso-transmi-one.vercel.app |
| Workflows | https://github.com/kevin-nieto-callejas/pulso-transmi/actions |
| Reloj | `curl -s https://pulso-transmi.72-60-245-2.sslip.io/v1/clock` |
| Ciclo vigente | `curl -s https://pulso-transmi.72-60-245-2.sslip.io/v1/forecast-cycles/current` |

---

## 1. Cuando el reloj arranque

El sistema lo detecta solo y **no depende del PC** (hallazgo #39):
`relay.yml` es un job que vive ~5 h 40 min en GitHub, consulta el ciclo cada
45 s y, cuando abre uno, corre ingesta, inferencia y evaluación. Al terminar
se relanza solo. Si la cadena se rompe, el cron `7,27,47` la revive.
`inference.yml` (cron cada 5 min, que GitHub en la práctica dispara cada 3-5 h)
queda de respaldo. El vigilante local `ops/nightwatch.ps1` ya no hace falta.

Para comprobar que la cadena está viva, tiene que haber una corrida de `relay`
`in_progress`:

```bash
gh run list -R kevin-nieto-callejas/pulso-transmi --workflow=relay.yml --limit 3
# Si no hay ninguna en curso:
gh workflow run relay.yml -R kevin-nieto-callejas/pulso-transmi
```

**Revisar en la primera hora, en este orden:**

1. **¿Llegaron datos nuevos?** En el dashboard, el recuadro "Recolector"
   debe mostrar el estado y las filas ingeridas de las corridas recientes.
2. **¿Se detectó el ciclo?** En Actions, el log de la corrida de `relay` debe
   tener un grupo `HH:MM:SS ciclo cyc_...` con `Submission ACEPTADA`.
3. **¿La entrega quedó aceptada?** En el portal, recuadro "Última entrega":
   estado **ACEPTADA** y cobertura 48 predicciones.
4. **¿Se está evaluando?** Cuando llegan los valores reales, el dashboard
   muestra accuracy por ciclo y estación; puede haber demora respecto al
   recibo porque la realidad se publica después.

Si los cuatro salen bien, deja que la automatización continúe y revisa de
nuevo tras la siguiente apertura o ante una alerta.

---

## 2. Qué es normal

Medido sobre 300 ciclos en la batería de esfuerzo. **Conviene leerlo antes
de alarmarse por un número feo.**

| Señal | Normal | Cuándo preocuparse |
|---|---|---|
| Accuracy de un ciclo | Dependía del régimen; 85-86 antes del drift | En las revisiones 3 y 4 puede caer mucho durante la transición; comparar ventanas equivalentes |
| Variación entre ciclos | ±2.6 puntos antes del drift | En drift fuerte manda la tendencia reciente y la cobertura, no el umbral histórico aislado |
| Duración de `inference` | ~9 min | Es la vigilancia de 8 min, no un cuelgue |
| Corridas del collector | ~4 por hora | Que falten algunas es esperable: GitHub descarta corridas |
| Alarmas de drift | ~0.4 por semana | Varias al día = algo cambió de verdad |

**El error más caro sigue siendo reaccionar a un ciclo aislado.** Un ciclo
tiene cuatro puntos por estación. En la fase de drift el profesor pidió un
chequeo explícito de los últimos seis ciclos contra 85%; `retrain-watch.yml`
automatiza ese chequeo y `train.py` solo promueve si el candidato supera al
champion bajo la misma validación temporal.

---

## 3. Problemas y qué hacer

### No se entregó un ciclo

**Primero:** mirar Actions. ¿Hay una corrida de `relay` en curso? Si no, lanzarla
(ver la sección 1). Después, ¿la corrida de `inference` se ejecutó?

- **No corrió** → GitHub pudo descartar o retrasar una corrida. Mientras el
  ciclo siga abierto, volver a consultar estado y recibo; si no hay aceptación,
  disparar `inference.yml` una vez. La concurrencia y la comprobación de
  idempotencia evitan duplicar un ciclo ya aceptado.
- **Corrió y falló** → abrir el log y buscar el error. Ver abajo.
- **Corrió y dijo "Sin ciclo abierto"** → puede ser que la ventana ya había
  cerrado. Verificar `closes_at` del ciclo.

**Entrega manual de urgencia** (solo si el ciclo sigue abierto):

```bash
python src/infer.py
```

Es idempotente: si ya se entregó ese ciclo con ese modelo, no reenvía ni
gasta intentos.

### `401 invalid_api_key`

La llave se revocó o rotó. Generar una nueva en el portal y actualizarla en
los dos sitios:

```bash
gh secret set PULSO_API_KEY --repo kevin-nieto-callejas/pulso-transmi --body "ptm_live_..."
# y en el .env local
```

### `409 attempt_limit_reached`

Se usaron los 3 intentos del ciclo. **No insistir.** Revisar por qué se
gastaron: normalmente significa que algo reenvió con contenido distinto.

### `422 invalid_target_set`

Los targets no coinciden con los que pidió la API. Nunca fabricarlos:
`infer.py` los toma tal cual del ciclo. Si aparece, es que el contrato
cambió — correr `python src/check_contract.py`.

### El modelo no carga

```bash
python src/check_model.py
```

Si falla, el champion está roto. Volver al anterior:

```bash
python src/rollback.py --listar
python src/rollback.py --auto --motivo "el champion no carga en Actions"
```

El rollback verifica que el destino cargue **antes** de cambiar nada, y
excluye los modelos de un solo horizonte aunque tengan mejor métrica.

### El recolector va atrasado

`infer.py` avisa en el log: `el ancla va N min por detras del data_cutoff`.

Costo medido: 30 min de atraso cuestan **2.2 puntos**; 60 min, **2.5**. A
partir de ahí se estanca. No es catastrófico, pero si es persistente hay que
mirar por qué el collector no está corriendo.

### Llega correo de `contract-watch`

El profesor cambió algo. **No es urgente y no significa que algo esté roto.**

```bash
python src/check_contract.py
```

Dice qué cambió. Si no afecta endpoints ni campos que usamos, actualizar
`contract/expected.json` con el SHA nuevo y listo. **Commitearlo y subirlo en el
mismo momento**: si la alarma se queda en rojo, el siguiente cambio de verdad
llega como un correo más, idéntico a los anteriores. Así pasó con el contrato
v2 (hallazgo #37): 5 días en rojo por un `expected.json` sin commitear.

Ojo: este vigilante no ve cambios en el formato de las observaciones, porque
la API no publica ese esquema. Esa falla se nota en el collector (`KeyError`
en `ingest.py`) y en el aviso `el ancla va N min por detras` de `infer.py`.

### El accuracy cae de verdad

Solo si la **media de 24 horas** cae, no un ciclo:

```bash
python src/evaluate.py
```

Da la decisión: mantener, investigar o reentrenar. **Hacerle caso.** Si dice
investigar, investigar; el umbral ya tiene en cuenta el ruido normal.

Si dice reentrenar:

```bash
python src/train.py
```

La regla de promoción decide sola si el candidato entra. Nunca promover a
mano.

---

## 4. Lo que NO hay que hacer

- **Reentrenar por un ciclo malo.** La variación normal entre ciclos es
  ±2.6 puntos.
- **Promover un modelo a mano** saltándose la regla de comparación.
- **Reenviar un ciclo ya entregado** cambiando el contenido: eso sí gasta
  intentos y puede generar conflicto de idempotencia.
- **Poner la `SERVICE_ROLE_KEY` o la `PULSO_API_KEY` en Vercel** ni en
  ningún sitio que llegue al navegador.
- **Tocar el sistema "por si acaso"** cuando todo está en verde.

---

## 5. Comandos útiles

```bash
# Estado general
curl -s https://pulso-transmi.72-60-245-2.sslip.io/v1/clock
curl -s https://pulso-transmi.72-60-245-2.sslip.io/v1/forecast-cycles/current

# Últimas corridas
gh run list --repo kevin-nieto-callejas/pulso-transmi --limit 10

# Log de una corrida
gh run view <ID> --repo kevin-nieto-callejas/pulso-transmi --log

# Salud del sistema
python src/check_model.py       # ¿carga el champion?
python src/check_contract.py    # ¿cambió algo del profesor?
python src/evaluate.py          # ¿hay drift? ¿qué decisión?

# Forzar una corrida
gh workflow run inference.yml --repo kevin-nieto-callejas/pulso-transmi
gh workflow run collector.yml --repo kevin-nieto-callejas/pulso-transmi
```

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

El sistema lo detecta solo: la inferencia corre en los minutos
`3,13,23,33,43,53` y cada corrida vigila 8 minutos más. No hay que
despertarlo.

**Revisar en la primera hora, en este orden:**

1. **¿Llegaron datos nuevos?** En el dashboard, el recuadro "Recolector"
   debe mostrar filas ingeridas mayores que cero. Hasta ahora todas las
   corridas decían `no_new_data` porque el stream estaba vacío.
2. **¿Se detectó el ciclo?** En Actions, la corrida de `inference` debe
   decir `Ciclo abierto: cyc_...` en vez de `Sin ciclo abierto`.
3. **¿La entrega quedó aceptada?** En el portal, recuadro "Última entrega":
   estado **ACEPTADA** y cobertura 48 predicciones.
4. **¿Se está evaluando?** Una o dos horas después, el dashboard debe
   empezar a mostrar accuracy por ciclo.

Si los cuatro salen bien, no hay nada más que hacer en todo el día.

---

## 2. Qué es normal

Medido sobre 300 ciclos en la batería de esfuerzo. **Conviene leerlo antes
de alarmarse por un número feo.**

| Señal | Normal | Cuándo preocuparse |
|---|---|---|
| Accuracy de un ciclo | 85-86 de media | Un ciclo suelto en 79 es normal (p05). El mínimo medido fue 56 |
| Variación entre ciclos | ±2.6 puntos | Solo importa si la **media de 24 h** cae |
| Duración de `inference` | ~9 min | Es la vigilancia de 8 min, no un cuelgue |
| Corridas del collector | ~4 por hora | Que falten algunas es esperable: GitHub descarta corridas |
| Alarmas de drift | ~0.4 por semana | Varias al día = algo cambió de verdad |

**El error más caro que se puede cometer el lunes es reentrenar por un ciclo
malo.** Un ciclo tiene 4 puntos por estación; con tan poca muestra, bajar a
79 no significa nada. Por eso el detector mira ventanas de 24 horas.

---

## 3. Problemas y qué hacer

### No se entregó un ciclo

**Primero:** mirar Actions. ¿La corrida de `inference` se ejecutó?

- **No corrió** → GitHub descartó la corrida (pasa; por eso hay 6 intentos
  por hora). Si se perdió el ciclo, ya no se recupera: **no hacer nada**, el
  siguiente ciclo entra solo.
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
`contract/expected.json` con el SHA nuevo y listo.

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

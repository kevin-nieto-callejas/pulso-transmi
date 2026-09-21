# Traspaso de contexto — Pulso TransMi

Documento para poner al día a una sesión nueva de Claude Code sobre el estado
del proyecto. Corte: 21 de septiembre de 2026.

---

## 1. Qué es esto

Reto MLOps **Pulso TransMi** (Universidad Externado, MLOps · Ciencia de
Datos, docente Julián Zuluaga). Pronosticar demanda de pasajeros en 12
estaciones de TransMilenio cada 15 minutos, y **operar** el sistema de forma
continua: recolectar, entrenar, predecir, entregar, evaluar y reaccionar al
drift.

**Estudiante:** Kevin Nieto — Grupo A · VIS2-2026II.

**Lo que se califica no es el accuracy**, sino sostener el ciclo completo. La
guía lo dice: *"El mejor proyecto no necesariamente tendrá la mejor
predicción en todos los ciclos. Será aquel que combine desempeño,
trazabilidad, capacidad de recuperación y decisiones justificadas."*

---

## 2. Enlaces

### Nuestros
| Qué | Dónde |
|---|---|
| Repositorio | https://github.com/kevin-nieto-callejas/pulso-transmi |
| Dashboard desplegado | https://pulso-transmi-one.vercel.app |
| GitHub Actions | https://github.com/kevin-nieto-callejas/pulso-transmi/actions |
| Supabase (requiere login) | https://supabase.com/dashboard/project/bawwhejgcvlawfqualrj |

### Documentos del repo
| Documento | Para qué |
|---|---|
| [`README.md`](../README.md) | Estado, arquitectura, decisiones, cómo reproducir |
| [`docs/BITACORA.md`](BITACORA.md) | Cronología y los problemas encontrados |
| [`docs/RUNBOOK.md`](RUNBOOK.md) | Qué hacer el día de competencia |
| [`docs/INFORME_FINAL.md`](INFORME_FINAL.md) | Entregable 11 de la guía |
| [`docs/entity-relation.md`](entity-relation.md) | Esquema y diagrama E-R |
| [`eda/EDA_REPORT.md`](../eda/EDA_REPORT.md) | Análisis exploratorio |
| [`.claude/skills/enviar-predicciones/`](../.claude/skills/enviar-predicciones/SKILL.md) | Cómo enviar predicciones |

### Del profesor
| Qué | Dónde |
|---|---|
| Portal del estudiante | https://pulso-transmi.72-60-245-2.sslip.io/ |
| API | https://pulso-transmi.72-60-245-2.sslip.io/docs |
| Repo del contrato | https://github.com/uexternadojz/pulso-transmi |
| Starter kit (remoto `upstream`) | https://github.com/uexternadojz/pulso-transmi-sdk |
| Guía metodológica (PDF) | `C:\Users\ASUS\Downloads\pulso-transmi-guia-metodologica-v1.0.pdf` |

### Datos en vivo (llave pública de solo lectura, segura de compartir)
```
https://bawwhejgcvlawfqualrj.supabase.co/rest/v1/model_versions?select=version_id,status,validation_metric&apikey=sb_publishable_b4hvWnqi4Sf_6OlOQ_27xw_KY7pgFaD
https://bawwhejgcvlawfqualrj.supabase.co/rest/v1/collector_runs?select=started_at,status,rows_ingested&order=started_at.desc&apikey=sb_publishable_b4hvWnqi4Sf_6OlOQ_27xw_KY7pgFaD
https://bawwhejgcvlawfqualrj.supabase.co/rest/v1/observations?select=count&apikey=sb_publishable_b4hvWnqi4Sf_6OlOQ_27xw_KY7pgFaD
```

---

## 3. Lo último (21/09, tarde)

Barrido de **657 experimentos** (`src/sweep.py`) sobre las tres familias de
boosting. Hallazgo principal, confirmado con el protocolo oficial de 5
cortes vía `src/revalidar.py`:

| Conjunto de features | n | Accuracy |
|---|---:|---:|
| Extendido **completo** | 49 | 86.29 |
| Extendido **sin lags semanales** | 45 | **86.76** |
| Solo base sin lag semanal | 25 | **86.79** |
| Champion vigente (ensamble) | 49 | 86.61 |

**Quitar las cuatro features de estacionalidad semanal sube 0.47 puntos.**
La feature que el EDA coronó como la más importante (`lag_672`, 91.8% de
importancia) es la que más le estorba a CatBoost. Lo que se medía como
"brecha de fragilidad" (2.11 puntos) era en realidad el costo de
sobreajustarse a ella.

Consecuencia operativa: un CatBoost solo (6 MB) le gana al ensamble de tres
modelos (38 MB). Menos dependencias en producción y menos descarga por
ciclo.

`catboost_sin_semanal` ya está agregado a `build_candidates()` en
`train.py`; la regla de promoción decide si entra.

## 4. Estado a 21/09

| | |
|---|---|
| **Reloj de competencia** | `waiting` — no ha empezado |
| **Champion** | `ensamble_extendido-20260920T223526Z`, accuracy **86.61** |
| **Tests** | 36, todos en verde |
| **Experimentos** | 657 en MLflow + 12 candidatos oficiales |
| **Commits** | 41 |
| **Entregables obligatorios** | 10.5 de 11 |
| **Bonos** | 5 de 5 |

Datos en Supabase: 51.840 observaciones · 4.320 de contexto · 12 estaciones ·
46 corridas del collector · 5 versiones de modelo · 12 predicciones · 1
submission aceptada · 2 señales de drift.

**Entrega de práctica aceptada:** `sub_b64352e18de5486ead9478e66d57c523`
(`cyc_practice_20260918`, 12/12, `is_official: true`).

---

## 5. Arquitectura

```
API del profe → collector → Supabase → experimentos/modelo
     ↑              (15 min)                    ↓
     └────────── GitHub Actions ←──── predicciones + submission
                  (min 3,13,23,33,43,53)
```

| Script | Qué hace | Cuándo |
|---|---|---|
| `src/ingest.py` | Recolecta desde el cursor confirmado | cron `8,23,38,53` |
| `src/infer.py` | Ciclo → 48 predicciones → entrega | cron `3,13,23,33,43,53` |
| `src/evaluate.py` | Evalúa, mide drift, decide | tras cada recolección |
| `src/train.py` | Compara 12 candidatos y promueve | manual, a propósito |
| `src/rollback.py` | Vuelve a un champion anterior | bajo demanda |
| `src/check_contract.py` | Avisa si el profe cambia algo | cron `17 6,12,18,23` |
| `src/check_model.py` | ¿el champion carga y predice? | con contract-watch |
| `src/sweep.py` | Barrido de experimentos con MLflow | manual |
| `src/revalidar.py` | Revalida los mejores con el protocolo oficial | manual |
| `src/simulate_cycle.py` | Simulacro de ciclo completo | manual |
| `src/stress_test.py` | Batería de esfuerzo | manual |

**Workflows:** `ci.yml`, `collector.yml`, `inference.yml`, `contract-watch.yml`.

---

## 6. Decisiones que no hay que deshacer

| Decisión | Por qué |
|---|---|
| **Horizonte directo** (`horizon_minutes` como feature, 4 filas por ancla) | Un modelo que predice su propia fila solo sirve para +15 min |
| **Split siempre temporal, por momento ancla** | Cada ancla genera 4 filas; partir por fila mete fuga de futuro |
| **Accuracy sin ponderar entre estaciones** | Es la fórmula oficial |
| **El horizonte se mide desde el ancla real, no desde `data_cutoff`** | Si el collector va atrasado, medir mal da predicciones malas sin error visible |
| **Paginar de 1000 en 1000** | PostgREST nunca devuelve más, sin importar el `limit` |
| **Caché del artefacto por versión** | El ensamble pesa 38 MB; sin caché son ~6 GB/semana |
| **Drift contra el propio historial, no contra la validación** | El accuracy por ciclo promedia 85.74; la validación dice 86.61 |
| **Referencia de drift sin solapar con la ventana actual** | Si no, a mayor caída menos detección |
| **Reentrenamiento manual** | Nunca se ha visto una degradación real que valide los umbrales |
| **La API key nunca llega al navegador** | Por eso el dashboard no muestra leaderboard |

---

## 7. Los problemas encontrados (material del informe)

1. **El modelo solo predecía un paso.** 86.77 parecía bueno, pero solo servía
   para +15 min. Corregido a horizonte directo: 85.22. Bajar el número fue lo
   correcto.
2. **Paginación de Supabase.** Cargaba 1 de 12 estaciones. Solo apareció con
   volumen real.
3. **Horizonte medido desde `data_cutoff`.** Falla silenciosa: con el
   collector atrasado habría predicho mal sin lanzar un error.
4. **GitHub descarta corridas programadas.** 2 de 28 esperadas en 7 horas.
5. **El monitoreo calibrado contra el número equivocado.** Y una conclusión
   propia corregida: se creyó que el accuracy caía 13 puntos de madrugada
   (eran 5 mediciones); con 2.229 ciclos resultó ser 1 punto.
6. **El detector de drift se tapaba a sí mismo.** Una caída de 10 puntos se
   detectaba menos (2%) que una de 6 (10%), porque la referencia incluía
   ventanas ya contaminadas.

**Patrón:** cuatro de los seis no lanzaban ningún error. Aparecieron midiendo
a escala, no probando casos sueltos.

---

## 8. Números de referencia (medidos, no estimados)

| Medición | Valor |
|---|---|
| Accuracy del champion (validación 5 cortes) | 86.61 |
| Por horizonte | 86.95 / 86.52 / 86.23 / 85.99 |
| Accuracy por ciclo suelto | media 85.74, desviación 2.60 |
| Peor ciclo de 2.229 | 56.43 |
| Percentil 5 | ~79 |
| Baseline ingenuo (lag 24 h) | 77.89 |
| Techo teórico | 88–89 |
| Brecha de fragilidad (sin lag semanal) | 2.11 puntos |
| Costo de 30 min de atraso del collector | −2.2 puntos |
| Falsas alarmas de drift | 0.4 por semana |
| Detección de caída de 6 / 10 puntos | 98% / 100% |
| Tiempo de armar 48 predicciones | 1.3 s |

---

## 9. Entorno de trabajo (detalles prácticos)

- **Ruta:** `\\wsl.localhost\Ubuntu\home\asus\code\clase3107\pulso-transmi-sdk`
  (ruta de red UNC, con sus rarezas).
- **Git:** siempre `git -c safe.directory='*' <cmd>` — hay "dubious ownership".
- **`gh` CLI:** no está en el PATH. Usar `"C:\Program Files\GitHub CLI\gh.exe"`,
  y antes exportar `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory
  GIT_CONFIG_VALUE_0='*'`.
- **Python:** `.venv/Scripts/python.exe` (venv de Windows).
- **Variables de entorno:** no se cargan solas. Con PowerShell:
  ```powershell
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
      [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
  }
  ```
  Con bash: `export $(grep -v '^#' .env | grep -v '^$' | xargs -d '\n')`
- **Supabase desde PowerShell:** `Invoke-RestMethod` falla con la clave
  secreta (Supabase lo toma por navegador). Usar `curl`.
- **MLflow:** la base está en `%LOCALAPPDATA%\pulso-transmi\mlflow.db`, no en
  el repo — SQLite no funciona sobre rutas de red.
- **Node no está instalado** (por si hace falta algún validador).
- **`TaskStop` no mata el proceso hijo.** Detener una tarea de fondo deja el
  Python corriendo: se comió 13 GB de RAM y tumbó las corridas siguientes sin
  dar ningún error. Hay que matarlo con `Stop-Process -Id <pid> -Force` y
  comprobar con `Get-Process python`.

### Secretos
**No están en este documento a propósito.**

| Secreto | Dónde vive |
|---|---|
| `SUPABASE_SERVICE_ROLE_KEY` | `.env` local (gitignored) + GitHub Secrets |
| `PULSO_API_KEY` | `.env` local + GitHub Secrets |
| `SUPABASE_KEY` (publicable) | `.env` y también en el dashboard, a propósito |

La llave publicable es de solo lectura: verificado, lectura 200 y escritura
401.

---

## 10. Qué falta

**Bloqueado por el profesor (el reloj sigue en `waiting`):**
- Evaluación de accuracy con ciclos reales — `prediction_evaluations` y
  `cycle_metrics` están vacías.
- Señal de drift de rendimiento real.
- Cierre del informe final con resultados de la competencia.

**En nuestras manos:** nada pendiente. Se llegó a rendimientos decrecientes:
más modelos y más features ya no mueven la aguja (estamos a ~3 puntos del
techo teórico, y probar objetivo MAE y pesos por estación no dio nada).

---

## 11. Cómo trabajar en este proyecto

Lo que ha funcionado y conviene mantener:

1. **Verificar en vivo, no de memoria.** Consultar la base, la API y los
   workflows antes de afirmar un estado.
2. **Desconfiar de las pruebas que pasan.** Cuatro de los seis bugs pasaban
   todos los controles. Preguntarse siempre si la verificación verifica algo:
   una corrida "probó" que el ensamble cargaba, pero terminaba antes de
   cargarlo.
3. **Medir a escala antes de concluir.** Una conclusión sacada de 5
   mediciones resultó falsa con 2.229.
4. **Elegir bien la métrica.** "10% de chequeos en alarma" sonaba inaceptable;
   eran 3 episodios en 43 días.
5. **Documentar los errores, no esconderlos.** Es lo que premia la rúbrica, y
   la sección más larga de la bitácora.
6. **No optimizar el accuracy a ciegas.** La guía advierte contra perseguir la
   métrica del histórico estático.

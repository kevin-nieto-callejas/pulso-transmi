# Bitácora del proyecto — Pulso TransMi

Registro cronológico de qué se construyó, qué se rompió, cómo se detectó y
qué se decidió. Es la materia prima del informe final (entregable 11:
*"qué cambió, qué funcionó y qué harían después"*) y se actualiza a medida
que avanza el proyecto, no al final.

**Equipo:** Kevin Nieto — Grupo A · VIS2-2026II
**Repositorio:** https://github.com/kevin-nieto-callejas/pulso-transmi
**Base de datos:** Supabase (proyecto `pulso-transmi`)

---

## Resumen de estado

| Fase | Estado |
|---|---|
| 1 · Comprender | Completa |
| 2 · Construir la memoria | Completa y automatizada |
| 3 · Experimentar | Completa (13 candidatos + baseline + 657 experimentos en MLflow) |
| 4 · Operar | Submission aceptada; workflows entregando solos |
| 5 · Aprender del error | Codigo completo y probado; falta evidencia con ciclos reales |

**Modelo champion vigente:** `catboost_sin_semanal-20260921T161638Z` —
accuracy **86.76** (validación temporal de 5 cortes, 4 horizontes). CatBoost
sobre 45 features: el conjunto extendido menos la estacionalidad semanal.
Salió del barrido, no de elegir a mano.

**Bonos:** los cinco de la guía están cubiertos.

---

## Línea de tiempo

### 16 de septiembre — Fases 1 y 2

- Descarga del histórico: 12 estaciones, 45 días, 51.840 observaciones,
  granularidad de 15 minutos.
- EDA completo: calidad de datos, estacionalidad, correlaciones, selección de
  features, validación cruzada temporal ([`eda/EDA_REPORT.md`](../eda/EDA_REPORT.md)).
- Esquema de 11 tablas en Supabase y migración del histórico
  ([`docs/entity-relation.md`](entity-relation.md)).
- Primer modelo champion registrado con su artefacto en Supabase Storage.

### 17 de septiembre — Fases 2 y 3

- Collector incremental idempotente (`src/ingest.py`), probado contra el stream
  real vacío.
- Comparación de 8 candidatos y promoción del champion con regla explícita.

### 18 de septiembre — Fase 4

- Automatización real en GitHub Actions (collector e inferencia).
- El docente publicó el portal de estudiantes y abrió la ronda de práctica
  (`cyc_practice_20260918`).
- **Submission aceptada:** `sub_b64352e18de5486ead9478e66d57c523`, 12/12
  predicciones, `is_official: true`.
- Vigilante del contrato del docente (`src/check_contract.py`).

### 20 de septiembre — Fase 5, bonos y endurecimiento

- El docente publicó el API 0.5.0 (rotación de llave y dashboard de cohorte);
  revisado: no afecta ningún endpoint ni campo que usemos.
- Ronda de mejora del modelo: 49 features —incluidos los pronósticos de clima
  que la API entregaba y nunca se usaban—, LightGBM, CatBoost y un ensamble.
  Champion nuevo con 86.61 frente a 85.22.
- Monitoreo, drift y decisión de reentrenamiento (`src/evaluate.py`).
- Estrategia de rollback (`src/rollback.py`), probada de punta a punta.
- Dashboard desplegado: https://pulso-transmi-one.vercel.app
- Simulacro de ciclo completo (`src/simulate_cycle.py`), que destapó el
  problema del monitoreo descrito abajo.

### 21 de septiembre — barrido masivo y champion nuevo

- Barrido de **657 experimentos** en MLflow sobre tres familias de boosting
  (`src/sweep.py`), con validación barata de 3 cortes.
- Las mejores configuraciones tenían algo en común inesperado: **ninguna
  usaba `lag_672`**, la feature que el EDA había coronado como la más
  importante (91.8%).
- `src/revalidar.py` volvió a medir las mejores con el protocolo oficial de
  5 cortes. Confirmado: quitar las cuatro features de estacionalidad semanal
  sube de 86.29 a 86.76 con el mismo modelo.
- Ese candidato se agregó a `src/train.py` y compitió contra los otros 12
  bajo la misma validación, sin promoverlo a mano. **Ganó: 86.76 contra
  86.61 del ensamble**, que pasó a `historical`.
- Efecto secundario: el artefacto bajó de 38 MB a 19 MB y la inferencia dejó
  de necesitar LightGBM y XGBoost en producción.
- Se corrigió la etiqueta "brecha de fragilidad" en el código y los
  documentos: medía otra cosa de la que decía (ver problema 7).
- Catálogo de hallazgos ([`HALLAZGOS.md`](HALLAZGOS.md)) para no volver a
  descubrir lo aprendido.

---

## Problemas encontrados y cómo se resolvieron

Esta sección es deliberadamente la más larga: los errores encontrados y
corregidos dicen más del proceso que los aciertos.

### 1. El modelo solo sabía predecir un paso adelante

**Síntoma:** ninguno — ese es el punto. El modelo reportaba 86.77 de accuracy
y parecía correcto.

**Problema real:** se entrenó para predecir la demanda de su propia fila
usando `lag_1` (el período anterior). Eso solo generaliza a +15 min. Para
+30/+45/+60 min ese mismo mecanismo exigiría conocer demanda posterior al
`data_cutoff`, es decir, justo lo que se está tratando de predecir. El modelo
habría fallado o entregado basura en 3 de cada 4 predicciones de cada ciclo.

**Detección:** al releer la guía a fondo antes de construir la inferencia, no
por una prueba que fallara.

**Solución:** estrategia de *horizonte directo*. Cada fila ancla se apila 4
veces (una por horizonte), se agrega `horizon_minutes` como feature y el
target es la demanda real desplazada ese horizonte hacia adelante, con lags
siempre anclados al momento de predicción.

**Consecuencia honesta:** el accuracy bajó de 86.77 a 85.22. No es una
regresión: el número anterior medía un problema más fácil que el real y nunca
habría aparecido en el leaderboard. El desglose por horizonte muestra el
patrón esperado (86.23 en +15 min → 84.59 en +60 min).

### 2. Supabase devuelve máximo 1.000 filas y la paginación asumía 5.000

**Síntoma:** la primera corrida contra un ciclo real falló con
`No hay historico reciente para la estacion '05000'`.

**Problema real:** PostgREST nunca devuelve más de 1.000 filas por respuesta,
sin importar el `limit` solicitado. La condición de corte de la paginación
(`len(page) < 5000`) se cumplía en la primera página, así que solo se cargaban
~1.000 filas que, ordenadas por estación, no alcanzaban ni para la segunda de
las 12.

**Por qué no se detectó antes:** todas las pruebas previas usaban conjuntos
pequeños o el stream vacío. El bug solo aparecía con volumen real.

**Verificación:** `limit=5000` devuelve 1.000 filas y la cabecera
`Content-Range: 0-999/51840`.

**Solución:** `fetch_all_rows()` pagina con cabeceras `Range` de 1.000 en
1.000 hasta agotar, con test de regresión.

### 3. El horizonte se medía desde el corte de datos, no desde el ancla

**Síntoma:** ninguno todavía — se detectó preparando la competencia real.

**Problema real:** las features salen de la última observación disponible (el
ancla), pero el horizonte se calculaba contra el `data_cutoff` del ciclo. En
la práctica coincidieron, así que funcionó. En competencia el `data_cutoff`
avanza cada hora y, si el collector va atrasado, el ancla queda antes del
corte: se le diría al modelo "salta 15 minutos" cuando en realidad debe saltar
45. Predicciones sistemáticamente malas **sin ningún error visible**.

**Solución:** el horizonte se mide contra el timestamp real del ancla; se
avisa cuando hay atraso y cuando el horizonte supera el máximo entrenado (60
min), señal de que el collector se quedó atrás. Además el workflow de
inferencia ejecuta el collector justo antes de inferir, con
`continue-on-error` (entregar con datos algo viejos es mejor que no entregar).

### 4. GitHub descarta las corridas programadas

**Síntoma:** de ~28 corridas programadas esperadas en 7 horas, solo se
ejecutaron 2.

**Diagnóstico:** no era cuota (repositorio público, minutos ilimitados), ni un
fork, ni workflows desactivados. GitHub no encola las corridas atrasadas: las
descarta bajo carga, y `:00` y `:30` —donde estaba el cron del collector— son
los minutos más congestionados de la plataforma.

**Por qué importa:** perder la corrida de un ciclo significa perder su ventana
de 25 minutos, es decir 48 predicciones en cero. Corresponde a la tercera
señal de la guía: *"falla operacional: el pipeline no produjo o envió
resultados — corregir la operación antes de culpar al modelo"*.

**Solución en dos capas:**
1. Más intentos en minutos menos congestionados (inferencia cada 10 min,
   collector cada 15, nunca `:00` ni `:30`).
2. Cada corrida vigila 8 minutos adicionales revisando cada 2, para que una
   sola corrida que sí arranque cubra buena parte de la ventana. Un fallo
   pasajero de red se reintenta en vez de abortar.

Reenviar no duplica ni gasta intentos: la llave de idempotencia es estable por
(ciclo, modelo) y, antes de rearmar el batch, se consulta si ese ciclo ya fue
entregado.

### 5. El monitoreo se calibró contra el número equivocado

**Síntoma:** ninguno. Todo en verde.

**Cómo se detectó:** un simulacro de ciclo completo (`src/simulate_cycle.py`)
creado para otra cosa —el pipeline de evaluación nunca había escrito una fila
real en `prediction_evaluations` ni en `cycle_metrics`—.

**Primera conclusión, equivocada:** el simulacro midió 72.17 en un ciclo
nocturno contra 85.73 en uno de mediodía, y se concluyó que el modelo rendía
peor de noche. Estaba sacado de **cinco mediciones**.

**Lo que dicen 2.229 ciclos:** el efecto de la hora es de un punto
(madrugada 85.10, tarde 86.29). Lo que existe es ruido de ciclo: desviación
2.60, con mínimos de 57.79, porque cada ciclo se evalúa con cuatro puntos por
estación. El 72.17 era el 0.6% peor por azar, no un patrón.

**Problemas reales que sí destapó:**

1. *Punto de comparación equivocado.* Se comparaba contra la métrica de
   validación (86.61), que agrupa todas las predicciones, cuando el accuracy
   por ciclo promedia 85.74. Sesgo de 0.87 puntos hacia la falsa alarma.
2. *Umbral por debajo del ruido.* Dos puntos sobre una ventana de 24 ciclos
   son 1.3 desviaciones: alarma por azar el 10% de las veces.

**Solución:** comparar la ventana reciente contra la mediana de sus propias
ventanas anteriores, con umbral de tres desviaciones medidas sobre ese mismo
historial, y usando mediana en vez de promedio para que un ciclo catastrófico
no arrastre la ventana.

**Limitación documentada:** tres desviaciones sobre 24 horas son 4.5 puntos.
Una degradación de la escala de 2.11 puntos —la que sufre Random Forest al
perder la estacionalidad semanal, usada aquí solo como orden de magnitud— no
la ve este detector. Hacen falta ventanas más largas.

**Lo que enseñó:** dos cosas. Que un accuracy por ciclo y uno de validación
no son comparables aunque se llamen igual. Y que cinco mediciones no son una
conclusión — el propio error de interpretación se corrigió midiendo más.

### 6. El detector de drift se tapaba a sí mismo

**Síntoma:** ninguno; los 32 tests que existían entonces pasaban.

**Cómo se detectó:** una batería de pruebas de esfuerzo
(`src/stress_test.py`) que mide sensibilidad y falsas alarmas a escala, en
vez de comprobar casos sueltos.

**Problema real:** una caída de 10 puntos se detectaba el **2%** de las
veces, menos que una de 6 (**10%**). Que a mayor degradación hubiera menos
detección delataba un error lógico: el periodo de referencia incluía
ventanas que ya contenían la caída. Cuanto mayor el bajón, más se inflaba la
desviación de referencia, más subía el umbral, y más se tapaba a sí mismo.

**Solución:** la referencia termina antes de que empiece la ventana actual, y
la desviación se estima con bloques que no se solapan —ventanas corridas
hora a hora comparten 23 de sus 24 ciclos, así que subestiman la
variabilidad real—.

**Resultado, medido sobre 100 historiales por tamaño:**

| Caída | Detectada |
|---|---:|
| 2 puntos | 26% |
| 4 puntos | 78% |
| 6 puntos | 98% |
| 10 puntos | 100% |

Con 0.4 falsas alarmas por semana sobre historiales sanos.

**Un detalle de método:** la primera medición reportó 8-16% de falsas
alarmas y parecía inaceptable, hasta contar **episodios** en vez de
chequeos. Las 101 alarmas de una serie de 43 días resultaron ser 3
episodios: una misma excursión dispara muchos chequeos seguidos. La métrica
que importa es cada cuánto alguien tiene que ir a mirar.

### 7. La métrica se llamaba "fragilidad" y medía otra cosa

**Síntoma:** el entrenamiento imprimía en cada corrida `Brecha de fragilidad
(rf_full - rf_no_weekly_lag): 2.11 puntos`, y tanto el README como el informe
lo presentaban como *el riesgo principal del proyecto*: si `lag_672` dejaba de
ser confiable por drift, perderíamos 2.11 puntos.

**Qué pasaba:** el número es correcto, pero no significa lo que decía la
etiqueta. Mide lo que pierde **Random Forest con el conjunto base** al
quitarle la estacionalidad semanal — una propiedad de ese modelo, no del
dato. El barrido mostró que CatBoost **mejora** con las mismas cuatro
features fuera (86.29 → 86.76).

**Por qué importa:** los dos resultados son ciertos a la vez y eso es lo
interesante. Random Forest se apoya en `lag_672` y sufre sin él; CatBoost se
sobreajusta a él y rinde más sin él. Lo que medíamos como "fragilidad del
dato ante drift" era en realidad **el costo de sobreajustarse a una señal**.

**Solución:** se renombró en el código (`dependencia_rf`, y en
`src/evaluate.py` la constante pasó a `DEGRADACION_DE_REFERENCIA`) y se
corrigió la narrativa en README, informe y handoff. Como el champion vigente
ya no usa ninguna feature semanal, ese drift dejó de ser un riesgo activo.

**Lo que enseñó:** una etiqueta equivocada sobrevive más que un bug, porque
nada falla. El número se imprimió correctamente durante días mientras la
frase que lo acompañaba decía lo contrario de lo que el dato sostenía.

### 8. Otros arreglos menores

- Índice único parcial `one_champion_only` para que la base impida a nivel
  físico tener dos champions simultáneos.
- Índices faltantes en llaves foráneas señalados por el linter de Supabase.
- Sintaxis inválida (`PK_FK`) que impedía renderizar el diagrama Mermaid.
- Petición redundante que generaba un `400` en cada corrida al recrear el
  bucket de Storage.
- CI instalaba solo el extra `dev`, sin `ml`, y no podía importar los módulos
  de inferencia.

---

## Decisiones y por qué

| Decisión | Razón |
|---|---|
| Partición siempre temporal (`TimeSeriesSplit`), nunca aleatoria | Mezclar futuro y pasado da métricas optimistas que no representan la competencia. |
| Horizonte directo (un modelo con `horizon_minutes`) en vez de recursivo | Evita acumular error encadenando 4 predicciones de un paso y elimina por diseño la fuga de futuro. |
| Corte de la validación por momento ancla, no por fila | Cada ancla genera 4 filas; partir por fila dejaría horizontes del mismo instante a ambos lados del corte. |
| Accuracy promediado sin ponderar entre las 12 estaciones | Es la fórmula oficial: una estación de gran volumen no puede ocultar el mal desempeño de una pequeña. |
| RLS con lectura pública y escritura solo con `service_role` | El esquema no contiene datos personales; permite que el dashboard y el docente consulten sin exponer escritura. |
| Artefactos en Supabase Storage, no en el repositorio | *"Un resultado que solo funciona en el computador de una persona no está terminado"*. Un binario de 1.9 MB tampoco pertenece a git. |
| Reemplazo del champion sin comparar números incompatibles | El champion anterior midió un problema más fácil (un horizonte). Comparar 86.77 contra 85.22 habría descartado un modelo mejor. Se detecta revisando si `horizon_minutes` está entre sus features. |

---

## Estado de la evidencia

| Evidencia | Dónde |
|---|---|
| Análisis exploratorio | [`eda/EDA_REPORT.md`](../eda/EDA_REPORT.md) |
| Esquema y diagrama entidad-relación | [`docs/entity-relation.md`](entity-relation.md) |
| Comparación de candidatos | [`eda/reports/training_summary.json`](../eda/reports/training_summary.json) |
| Versiones de modelo y promociones | Tabla `model_versions` en Supabase |
| Bitácora de ingestas | Tabla `collector_runs` |
| Predicciones emitidas y recibo | Tablas `predictions` y `submissions` |
| Corridas automáticas | GitHub Actions del repositorio |

---

## Qué falta

- **Evidencia de evaluación con datos reales:** el código está completo y
  ejercitado de punta a punta con un simulacro, pero la ronda de práctica
  pidió un instante cuyo valor real nunca se publicó, así que
  `prediction_evaluations` y `cycle_metrics` siguen vacías. Depende de que
  arranque la competencia.
- **Cierre del informe final** con los resultados reales de la ventana
  competitiva. El documento ya está escrito: [`INFORME_FINAL.md`](INFORME_FINAL.md).

Los cinco bonos de la guía están cubiertos: dashboard en Vercel, MLflow,
pruebas automatizadas, estrategia de rollback y monitoreo de drift más allá
de la métrica de desempeño.

---

## Verificaciones hechas sobre el sistema

Registradas porque la guía valora poder demostrar el estado, no afirmarlo:

- Integridad: 51.840 observaciones = 12 estaciones × 4.320 períodos, sin
  duplicados ni valores negativos.
- Idempotencia: ejecuciones repetidas del collector y de la migración no
  alteran los conteos.
- Seguridad: la llave pública lee (HTTP 200) y no puede escribir (HTTP 401);
  cero alertas del linter de seguridad de Supabase; ningún secreto en el
  historial del repositorio.
- Reproducibilidad: el artefacto del champion se descarga desde Storage y
  vuelve a predecir en un entorno limpio (así opera GitHub Actions).
- Pruebas: 36 tests automatizados en CI, incluidos casos de regresión de cada
  error descrito arriba.
- Ciclo completo: un simulacro con 48 targets y cuatro horizontes recorre
  predicción, emparejamiento con la realidad, métricas y limpieza, y verifica
  el resultado con un cálculo independiente.
- Esfuerzo (`src/stress_test.py`): 300 ciclos repartidos por hora y día,
  impacto medido del recolector atrasado, sensibilidad y falsas alarmas del
  detector de drift, robustez ante entradas inválidas, y tiempos contra la
  ventana de entrega.

# Informe final — Pulso TransMi

**Kevin Nieto · Grupo A · VIS2-2026II**
MLOps · Ciencia de Datos · Universidad Externado de Colombia

Repositorio: https://github.com/kevin-nieto-callejas/pulso-transmi
Dashboard: https://pulso-transmi-one.vercel.app

> Actualización operativa: 1 de octubre de 2026. La fase competitiva sigue
> activa y cierra el 2 de octubre a las 23:59, hora de Bogotá. Los números de
> leaderboard son una fotografía al momento indicado; se actualizará el cierre
> final después del último ciclo.

---

## 1. Qué se construyó

Un sistema que recolecta, entrena, predice, entrega, evalúa y decide, corriendo
solo. No es un modelo con un pipeline alrededor: es un ciclo operativo donde el
modelo es una pieza reemplazable.

| Componente | Qué hace | Cadencia |
|---|---|---|
| `ingest.py` | Recolecta desde el cursor confirmado, idempotente | cada 15 min |
| `train.py` | Compara candidatos y promueve con regla explícita | manual, por decisión |
| `infer.py` | Consulta el ciclo, arma 48 predicciones, entrega | cada 10 min |
| `evaluate.py` | Une predicción con realidad, mide, vigila, decide | tras cada recolección |
| `rollback.py` | Vuelve a un champion anterior verificado | bajo demanda |
| `check_contract.py` | Avisa si el docente cambia el contrato | 4 veces al día |

Estado medible al 1/10: 63.660 observaciones, 234 ciclos oficiales con
submission aceptada de 48/48 predicciones, cero ciclos omitidos y cero
duplicados, 44 versiones de modelo registradas y 70 pruebas automatizadas en
verde. Las cifras vivas se consultan en el
[dashboard](https://pulso-transmi-one.vercel.app).

## Resultados de la competencia al 1 de octubre

La auditoría de ciclos oficiales confirmó 234/234 entregas aceptadas con
cobertura de 100%. A las 23:35 hora de Bogotá, el leaderboard acumulado
marcaba puesto **6**, accuracy **75.31** y cobertura **100%**. Este score mide
la demanda bajo el drift activo; no es comparable directamente con el 86.76
de validación del champion sobre 45 días de histórico estático.

El escenario original agotó su horizonte el 28 de septiembre. La API del
profesor pasó a 0.8.0 y habilitó una continuación de drift controlado hasta el
2 de octubre a las 23:59 Bogotá. El contrato de submission y los campos
consumidos por el pipeline permanecieron compatibles. La revisión 3 mostró una
onda de aproximadamente cuatro horas: en el análisis de 231 ciclos, la capa
adaptativa por onda obtuvo 83.29 frente a 80.14 del pipeline anterior, sin
regresión observada en el régimen normal. Esto es evidencia retrospectiva del
periodo analizado, no una garantía para cada ciclo futuro.

Durante la fase, `retrain-watch.yml` evalúa el promedio de los seis últimos
ciclos oficiales contra el umbral docente de 85%. Si solicita reentrenar,
`train.py` solo promueve al candidato cuando supera al champion con la misma
validación temporal. El champion base `catboost_sin_semanal` conserva 86.76 de
validación; la adaptación de producción usa información reciente disponible
hasta el corte y no altera las observaciones históricas.


## Fase final (3 y 4 de octubre): evolución de la fuente

La guía de la fase final (`docs/fase-final.md` del profesor) pide registrar
el cambio detectado, su impacto, las decisiones de reparación, la primera
entrega recuperada y la evolución de cobertura y accuracy. El detalle está en
los hallazgos #37 a #39 y en la bitácora. En resumen:

| | |
|---|---|
| **Cambio 1: fuente** | Contrato de observación v2: `measurement.value` en texto (o `null` con `quality=missing`) en lugar de `demand`. La ingesta cayó con `KeyError` y la inferencia siguió entregando con datos de 3 h atrás, sin ningún error visible. |
| **Cambio 2: demanda** | Revisión 4 del drift: desaparece la onda de 4 h y aparecen tendencias lentas. La onda corta seguía activa y acertaba cerca de 0. |
| **Impacto** | Ciclo `20260920T130000Z`: 19.84 de accuracy; el `150000Z` también salió con datos 3 h atrás. Ciclos `12:00Z` y `14:00Z` sin entregar, porque el disparador puntual vivía en el PC y el PC estaba apagado. |
| **Reparación** | Ingesta v1+v2 que omite los faltantes (no son cero), re-ingesta de las 3 h perdidas; la onda corta exige acierto en las últimas 2 h y entra persistencia + ½ tendencia; `relay.yml` entrega desde la nube; `contract-watch` vuelve a verde con 0.9.0. |
| **Primera entrega recuperada** | Ciclo `20260920T160000Z`, 4-oct 02:50 UTC: el relay la entregó 30 s después de abrir, con datos al corte. |
| **Evidencia de la adaptación** | Backtest causal rev 4: mezcla champion+perfil 64.0, persistencia 72.1, persistencia+½ tendencia 72.6; pipeline completo desde las 13:15 virtuales: 78-83. |
| **Cobertura y accuracy** | 4-oct 02:22 UTC: 186/188 ciclos (98.9%), acumulado 76.26, puesto 6 de 32. |
| **Adaptación continua** | Con más horas de la revisión 4 apareció una onda lenta de ~5-6 h por estación (hallazgo #40). Se agregó una sinusoide ajustada por estación en cada ciclo: con 1 armónico desde las 07:58 UTC, con 2 armónicos desde las 14:21 y con peso 0.7 desde el ciclo de las 17:50. Ciclos oficiales: 73-79 antes de la onda larga; 81.68 en el primero con ella; 82.68 y 85.65 con 2 armónicos. |
| **Leaderboard** | 4-oct 16:43 UTC: acumulado 76.11, puesto 5 de 32 (desde el 6). |

**Entrenamientos.** En la fase final no se promovió ningún modelo nuevo y el
champion sigue siendo `catboost_sin_semanal-20260921T161638Z`. La adaptación a
la revisión 4 no es un reentrenamiento: es una regla causal que solo usa
observaciones anteriores al `data_cutoff` de cada ciclo, y se validó con
backtest sobre ciclos ya resueltos. `retrain-watch.yml` sí entrenó candidatos
el 2-oct (12 versiones, datos exportados de Supabase hasta su `data_cutoff`
respectivo). Tras el arreglo del contexto (#36), esos candidatos empezaron a
ver el drift en su validación: bajaron de 86.74 a ~83. Ninguno superó al
champion bajo la regla de promoción, así que todos quedaron como `candidate`.
Esa comparación tiene un límite que hay que decir: champion y candidatos se
validaron sobre ventanas distintas, y la de los candidatos incluye el drift.

---

## 2. Qué cambió durante el proyecto

Tres cambios de rumbo, todos por haber encontrado que estábamos resolviendo el
problema equivocado.

### 2.1. El modelo pasó de "nowcasting" a horizonte directo

La primera versión alcanzaba **86.77** de accuracy y parecía terminada. No lo
estaba: predecía la demanda de su propia fila usando el período anterior como
señal. Eso solo sirve para +15 minutos. Para +30/+45/+60 el mismo mecanismo
exigiría conocer demanda posterior al `data_cutoff` — justo lo que hay que
predecir. El modelo habría fallado en **tres de cada cuatro** predicciones de
cada ciclo.

Se rehizo apilando cada fila ancla cuatro veces, una por horizonte, con
`horizon_minutes` como feature y el objetivo desplazado hacia adelante.

El accuracy bajó a **85.22**. Ese descenso fue la parte más incómoda del
proyecto y la más instructiva: el número anterior no era mejor, era **menos
cierto**. Medía un problema más fácil que el real. Aceptar una cifra peor y
defenderla fue una decisión deliberada.

### 2.2. La comparación de modelos dejó de ser numérica pura

Al reemplazar el champion apareció un problema de método: el vigente tenía
86.77 y el nuevo 85.22, pero medían cosas distintas. Compararlos habría
descartado el modelo correcto.

`train.py` ahora detecta la incompatibilidad revisando si `horizon_minutes`
está entre las features del champion, y en ese caso reemplaza sin comparar
números. La regla de promoción se mantiene intacta para versiones comparables.

### 2.3. Las features resultaron más decisivas que la arquitectura

Medición propia, con la misma validación:

| Cambio | Ganancia |
|---|---:|
| Añadir features nuevas (mismo XGBoost) | **+0.93** |
| Cambiar XGBoost → LightGBM (mismas features) | +0.32 |
| Duplicar los árboles de XGBoost | **−0.68** |

Entre las features nuevas estaban `rain_forecast` y `temperature_forecast`,
que la API entregaba desde el principio y **nunca se habían usado**. Se
alimentaba al modelo el clima *observado* para predecir el futuro, cuando lo
correcto es el pronóstico.

Champion final: CatBoost sobre 45 features —el conjunto extendido menos la
estacionalidad semanal—, **86.76** de accuracy (baseline ingenuo: 77.89).
Reemplazó a un ensamble de tres familias de boosting que marcaba 86.61.

---

## 3. Qué funcionó

**Medir lo que se supone antes de necesitarlo.** Se entrenó a propósito un
candidato sin la estacionalidad semanal para saber cuánto se perdería si esa
señal fallara: **2.11 puntos** en Random Forest. El número sirvió de escala
para fijar el umbral de drift.

Y sirvió para algo que no se buscaba: cuando el barrido mostró que CatBoost
**mejora** sin esas mismas features, quedó claro que 2.11 no medía fragilidad
del dato sino el costo de sobreajustarse a una señal. Tener la medición hecha
es lo que permitió reinterpretarla en vez de descubrirla tarde.

**Exigir evidencia antes de reaccionar.** El monitoreo no actúa ante un ciclo
malo. Mira la mediana de las últimas 24 horas y la compara con la de su propio
historial: un aguacero, una hora punta atípica o un ciclo con mala suerte no
mueven una mediana de 24 valores, pero una degradación sostenida sí.

**Separar falla operacional de falla del modelo.** Si hay ciclos sin entregar,
el sistema dice "arregla el pipeline" en vez de proponer reentrenar. Un
accuracy bajo por no haber entregado no se corrige tocando el modelo.

**No borrar nada.** Ningún champion se elimina; queda como histórico con su
métrica y su artefacto. Eso hizo posible el rollback, que se probó de verdad:
volver atrás, verificar que el sistema servía el modelo anterior, y regresar.

**Escribir pruebas de los errores encontrados.** Cada fallo dejó un test de
regresión. Son 32; varios existen porque algo se rompió primero.

---

## 4. Qué no funcionó, y qué costó

### 4.1. Dos fallas que no producían ningún error

Las más peligrosas no se anunciaban.

**El horizonte se medía desde el corte de datos, no desde el ancla.** Las
features salen de la última observación disponible, pero la distancia al
objetivo se calculaba contra el `data_cutoff` del ciclo. Coincidían en la
práctica, así que todo pasó en verde. En competencia, con el recolector
atrasado, se le habría dicho al modelo "salta 15 minutos" cuando debía saltar
45: predicciones sistemáticamente malas, sin una sola excepción en los logs.

**Una prueba que no probaba nada.** Se lanzó una corrida de inferencia para
verificar que el ensamble cargaba en GitHub Actions. Pasó en verde — pero como
no había ciclo abierto, el script terminaba *antes* de llegar a cargar el
modelo. La verificación habría fallado solo el día que hubiera un ciclo real.
De ahí salió `check_model.py`, que comprueba la carga sin depender de que haya
ciclo.

### 4.2. La infraestructura gratuita no se comporta como promete

**GitHub descarta corridas programadas.** De unas 28 esperadas en siete horas,
se ejecutaron **2**. No era cuota ni configuración: bajo carga, GitHub no
encola las corridas atrasadas, las descarta, y `:00` y `:30` son los minutos
más congestionados. Perder una corrida en competencia significa perder una
ventana de 25 minutos completa. Se mitigó con más intentos en minutos menos
congestionados y con vigilancia interna de 8 minutos por corrida.

**El mejor modelo pesaba 38 MB.** Descargarlo en cada ciclo serían ~6 GB por
semana, por encima del plan gratuito de Supabase. Quedarse sin cuota a mitad
de competencia significa dejar de entregar, y ninguna mejora de accuracy
compensa eso. Se resolvió con caché por versión en vez de sacrificar el
modelo. El champion que terminó ganando pesa 19 MB, así que el problema se
redujo a la mitad por su cuenta — pero la caché sigue siendo lo que lo
vuelve sostenible.

### 4.3. Errores de volumen que las pruebas pequeñas no ven

**PostgREST nunca devuelve más de 1000 filas.** La paginación asumía páginas de
5000, así que cortaba en la primera y cargaba **1 de las 12 estaciones**. Todas
las pruebas previas usaban conjuntos pequeños; el fallo apareció en la primera
corrida contra un ciclo real.

**Las tablas vacías vuelven sin columnas.** El monitoreo reventaba justo en el
estado normal previo a la competencia.

### 4.4. El monitoreo se calibró contra el número equivocado

Un simulacro de ciclo completo —creado para ejercitar el pipeline de
evaluación, que nunca había escrito una fila real— midió 72.17 en un ciclo
nocturno frente a 85.73 en uno de mediodía. La conclusión inmediata fue que
el modelo rendía peor de noche y que el detector de drift daría falsas
alarmas cada madrugada.

**Esa conclusión era falsa, y lo fue por medir poco.** Estaba sacada de cinco
ciclos. Al repetir la medición sobre **2.229 ciclos** evaluados fuera de
muestra, el efecto de la hora resultó ser de apenas un punto:

| Franja | Accuracy |
|---|---:|
| Madrugada (0-5 h) | 85.10 |
| Mañana (6-11 h) | 86.11 |
| Tarde (12-17 h) | 86.29 |
| Noche (18-23 h) | 85.45 |

Lo que sí existe es **ruido de ciclo**: la desviación entre ciclos es 2.60 y
el peor baja a 57.79 sin que nada falle, porque un ciclo se evalúa con apenas
cuatro puntos por estación. El 72.17 no era un patrón nocturno: era el 0.6%
peor, por azar.

El episodio dejó dos correcciones reales en el detector:

**El punto de comparación estaba mal.** Se comparaba contra la métrica de
validación (86.61 en ese momento), que agrupa todas las predicciones de la partición,
mientras que el accuracy de un ciclo promedia 85.74. Son dos formas de
agregar lo mismo; compararlas mete un sesgo de 0.87 puntos que hace
sobre-disparar. Ahora se compara la ventana reciente contra la mediana de sus
propias ventanas anteriores: drift es cambio respecto a como venía, no
diferencia contra un número de entrenamiento.

**El umbral estaba por debajo del ruido.** Dos puntos sobre una ventana de 24
ciclos son 1.3 desviaciones: alarma por azar una de cada diez veces. El
umbral pasó a expresarse en desviaciones medidas sobre el propio historial
(tres), y la ventana usa mediana en vez de promedio, para que un solo ciclo
catastrófico no la arrastre.

Queda una limitación conocida y documentada: con ventanas de 24 horas, tres
desviaciones son 4.5 puntos. Una degradación del orden de 2 puntos no la
vería este detector; haría falta una ventana más larga, a cambio de
reaccionar más tarde.

### 4.5. Una intuición que resultó falsa

Entrenar un modelo especializado por horizonte parecía obviamente mejor.
Medido: **86.39** contra **86.93** del modelo único. El compartido gana porque
aprende de cuatro veces más datos. Se descartó la idea.

---

## 5. Una trampa que vale la pena señalar

Al construir el rollback, la opción automática debía elegir "el histórico con
mejor métrica". Aplicado sin criterio, habría escogido esto:

```
xgboost_station-20260917   86.77   ← un solo horizonte
xgboost_station-20260918   85.22   ← multi-horizonte
```

El de mejor número solo sabe predecir +15 minutos. Un rollback "al más alto"
habría dejado el sistema entregando valores sin sentido en tres de cada cuatro
predicciones, en el peor momento posible: cuando ya algo había fallado.

El script excluye explícitamente los modelos incompatibles. Esa protección
existe únicamente porque ese error ya se había cometido antes en el proyecto.

---

## 6. Limitaciones honestas

- **El resultado de competencia sigue sujeto al cierre.** Los ciclos ya se
  evalúan con datos reales, pero el escenario continúa y el leaderboard puede
  cambiar hasta el último ciclo.
- **86.76 se midió sobre 45 días de histórico estático.** No hay drift en esos
  datos. El número puede no sostenerse cuando la ciudad cambie.
- **El reentrenamiento condicionado sí está automatizado durante esta fase.**
  El chequeo usa seis ciclos y el umbral explícito del profesor; la promoción
  permanece protegida por la comparación temporal del entrenamiento.
- **Los umbrales de drift están justificados, no validados.** Salen de
  mediciones propias, pero ninguno se ha enfrentado todavía a un drift real.
  Sí se validó, en cambio, que **no disparen cuando no deben**: el simulacro
  confirmó que la variación normal por hora del día no los activa.
- **El accuracy de un ciclo suelto no es comparable con el de validación.**
  El primero promedia 85.74 con desviación 2.60 (mínimo medido: 57.79); el
  segundo, 86.76, se calcula agrupando toda la partición. Un ciclo malo
  aislado no dice nada.
- **Degradaciones menores de ~4.5 puntos no las ve el detector** con ventanas
  de 24 horas. Ampliar la ventana las haría visibles a cambio de reaccionar
  más tarde; se dejó así porque reaccionar tarde a un drift pequeño es
  preferible a reaccionar en falso a uno inexistente.
- **La dependencia de `lag_672` dejó de ser el riesgo principal.** Se la
  señalaba como tal porque el EDA le daba 91.8% de importancia y quitarla le
  costaba 2.11 puntos a Random Forest. Medido con 657 experimentos, resultó
  al revés: CatBoost rinde **más** sin ella, y el champion vigente no usa
  ninguna feature de estacionalidad semanal. Ese drift concreto ya no nos
  afecta — lo que queda es la advertencia metodológica de que la importancia
  reportada mide uso, no aporte.

---

## 7. Qué haríamos después

**Al cierre de la fase, 2 de octubre**
1. Esperar la evaluación del último ciclo y capturar leaderboard final.
2. Auditar conteo final de ciclos, aceptaciones, cobertura y duplicados.
3. Registrar el champion vigente y el resultado de la última revisión de
   reentrenamiento.
4. Actualizar este informe con timestamp y conservar la diferencia entre
   validation accuracy y accuracy de competencia.

**A mediano plazo**
5. Ventana móvil de entrenamiento: que el modelo olvide datos viejos cuando el
   patrón cambie, en vez de promediarlos con los nuevos.
6. Predicción por intervalos en vez de valor único, para saber cuándo el modelo
   está inseguro — más útil operativamente que un número seco.

**Si hubiera más tiempo**
7. Modelos por estación para las que más se desvíen del comportamiento común.
8. ~~Sustituir el ensamble por un solo modelo si la ganancia de 0.14 puntos
   no justifica triplicar las dependencias en producción.~~ **Hecho:** el
   barrido produjo un CatBoost que además de ser más simple es más preciso
   (86.76 contra 86.61), así que la disyuntiva desapareció.

---

## 8. Lo que se aprendió

Que el número no es el proyecto. La decisión más difícil fue aceptar bajar de
86.77 a 85.22, y fue la correcta: la primera cifra medía un problema que nadie
nos iba a pedir resolver.

Que los fallos que importan no hacen ruido. Los peores errores del proyecto
—el modelo de un solo horizonte, el horizonte mal medido, la prueba que no
probaba nada y el monitoreo que habría gritado cada madrugada— habrían pasado
todos los controles en verde. Ninguno lanzaba una excepción. Los tests
pasaban. Solo aparecieron al ejercitar el sistema con datos reales, y tres de
los cuatro se encontraron buscando otra cosa.

De ahí la costumbre que más rindió: no dar por buena una verificación sin
comprobar que realmente verifica algo. La corrida que "probaba" el ensamble
terminaba antes de cargarlo; el simulacro que se hizo para validar el
pipeline de evaluación fue el que destapó el problema del monitoreo.

Y que en un sistema que debe operar solo, la fiabilidad vale más que la
precisión. Un modelo de 86.76 que entrega puntualmente vence a uno de 90 que
se queda sin cuota de almacenamiento el martes. Que el champion final sea
además el más liviano —19 MB contra los 38 del ensamble, una dependencia en
vez de tres— fue coincidencia afortunada, no diseño.

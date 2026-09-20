# Informe final — Pulso TransMi

**Kevin Nieto · Grupo A · VIS2-2026II**
MLOps · Ciencia de Datos · Universidad Externado de Colombia

Repositorio: https://github.com/kevin-nieto-callejas/pulso-transmi
Dashboard: https://pulso-transmi-one.vercel.app

> Este informe cubre lo construido hasta el inicio de la ventana competitiva.
> Los resultados de la competencia en vivo se añadirán al cierre; lo que sigue
> es lo que puede afirmarse con evidencia hoy.

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

Estado medible al cierre de esta etapa: 51.840 observaciones migradas sin
duplicados, más de 40 corridas del recolector registradas y creciendo, 5
versiones de modelo, una submission aceptada oficialmente, 30 pruebas
automatizadas en verde. Las cifras vivas se pueden consultar en el
[dashboard](https://pulso-transmi-one.vercel.app), no hace falta creerle a
este documento.

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

Champion final: ensamble de LightGBM + XGBoost + CatBoost sobre 49 features,
**86.61** de accuracy (baseline ingenuo: 77.89).

---

## 3. Qué funcionó

**Medir la fragilidad antes de necesitarlo.** Se entrenó a propósito un
candidato sin la estacionalidad semanal para saber cuánto se perdería si esa
señal fallara: **2.11 puntos**. Ese número dejó de ser curiosidad cuando hubo
que fijar el umbral de drift — es la magnitud de una degradación que rompe una
feature central, así que por debajo de eso es ruido.

**Exigir persistencia antes de reaccionar.** El monitoreo no actúa ante un
ciclo malo; pide tres seguidos. Un aguacero o una hora punta atípica no
justifican reentrenar.

**Separar falla operacional de falla del modelo.** Si hay ciclos sin entregar,
el sistema dice "arregla el pipeline" en vez de proponer reentrenar. Un
accuracy bajo por no haber entregado no se corrige tocando el modelo.

**No borrar nada.** Ningún champion se elimina; queda como histórico con su
métrica y su artefacto. Eso hizo posible el rollback, que se probó de verdad:
volver atrás, verificar que el sistema servía el modelo anterior, y regresar.

**Escribir pruebas de los errores encontrados.** Cada fallo dejó un test de
regresión. Son 30; varios existen porque algo se rompió primero.

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
compensa eso. Se resolvió con caché por versión en vez de sacrificar el modelo.

### 4.3. Errores de volumen que las pruebas pequeñas no ven

**PostgREST nunca devuelve más de 1000 filas.** La paginación asumía páginas de
5000, así que cortaba en la primera y cargaba **1 de las 12 estaciones**. Todas
las pruebas previas usaban conjuntos pequeños; el fallo apareció en la primera
corrida contra un ciclo real.

**Las tablas vacías vuelven sin columnas.** El monitoreo reventaba justo en el
estado normal previo a la competencia.

### 4.4. Una intuición que resultó falsa

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

- **No hay evaluación con datos reales.** La ronda de práctica pedía
  predicciones para un instante cuyo valor real nunca se publicó, así que las
  12 predicciones emitidas no son evaluables. El código de evaluación está
  escrito y probado con datos sintéticos, pero `prediction_evaluations` y
  `cycle_metrics` están vacías.
- **86.61 se midió sobre 45 días de histórico estático.** No hay drift en esos
  datos. El número puede no sostenerse cuando la ciudad cambie.
- **El reentrenamiento no está automatizado, a propósito.** El sistema decide
  *si* conviene reentrenar; ejecutarlo sigue siendo manual. Automatizar la
  promoción sin haber visto nunca una degradación real sería confiar en
  umbrales que no se han puesto a prueba.
- **Los umbrales de drift están justificados, no validados.** Salen de
  mediciones propias, pero ninguno se ha enfrentado todavía a un drift real.
- **La dependencia de `lag_672` sigue siendo el riesgo principal.** Cuesta 2.11
  puntos si esa señal falla, y es exactamente el tipo de patrón que un cambio
  de comportamiento urbano rompería.

---

## 7. Qué haríamos después

**Inmediato, cuando la competencia genere datos**
1. Confirmar que la evaluación mide lo que se espera en los primeros ciclos.
2. Revisar si los umbrales de drift disparan cuando deben, y ajustarlos con
   evidencia en vez de con razonamiento.
3. Vigilar la brecha entre el accuracy de validación y el real: si es grande,
   la validación temporal está siendo optimista.

**A mediano plazo**
4. Reentrenamiento automático, **solo después** de haber visto al menos una
   degradación real y comprobado que la decisión habría sido correcta.
5. Ventana móvil de entrenamiento: que el modelo olvide datos viejos cuando el
   patrón cambie, en vez de promediarlos con los nuevos.
6. Predicción por intervalos en vez de valor único, para saber cuándo el modelo
   está inseguro — más útil operativamente que un número seco.

**Si hubiera más tiempo**
7. Modelos por estación para las que más se desvíen del comportamiento común.
8. Sustituir el ensamble por un solo modelo si la ganancia de 0.14 puntos no
   justifica triplicar las dependencias en producción.

---

## 8. Lo que se aprendió

Que el número no es el proyecto. La decisión más difícil fue aceptar bajar de
86.77 a 85.22, y fue la correcta: la primera cifra medía un problema que nadie
nos iba a pedir resolver.

Que los fallos que importan no hacen ruido. Los tres peores errores —el modelo
de un solo horizonte, el horizonte mal medido y la prueba que no probaba
nada— habrían pasado todos los controles en verde y habrían costado la
competencia sin dejar rastro en ningún log.

Y que en un sistema que debe operar solo, la fiabilidad vale más que la
precisión. Un modelo de 86.61 que entrega puntualmente vence a uno de 90 que
se queda sin cuota de almacenamiento el martes.

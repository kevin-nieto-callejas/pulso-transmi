# Hallazgos — Pulso TransMi

Catálogo de lo que aprendimos midiendo, organizado por tema. A diferencia de
la [bitácora](BITACORA.md), que narra en orden lo que fue pasando, esto
agrupa las conclusiones para no tener que volver a descubrirlas.

Cada hallazgo dice **qué creíamos**, **qué medimos** y **qué implica**. Todos
tienen evidencia reproducible en el repo.

---

## Sobre el modelo

### 1. Predecir un paso no es predecir cuatro
**Creíamos:** un modelo con 86.77 de accuracy estaba listo.
**Medimos:** solo predecía la fila propia usando el período anterior. Para
+30/+45/+60 necesitaría datos posteriores al `data_cutoff`.
**Implica:** habría fallado en 3 de cada 4 predicciones de cada ciclo. Al
corregirlo a horizonte directo bajó a 85.22 — el número anterior medía un
problema más fácil que el real.

### 2. Las features rinden más que la arquitectura
**Medimos**, con la misma validación:

| Cambio | Ganancia |
|---|---:|
| Añadir features nuevas (mismo modelo) | **+0.93** |
| Cambiar XGBoost → LightGBM | +0.32 |
| Duplicar los árboles | **−0.68** |

**Implica:** ante la duda, buscar la señal que falta antes que el modelo más
grande. Duplicar el cómputo empeoró.

### 3. La feature "más importante" era la que más estorbaba
**Creíamos:** `lag_672` (demanda de hace una semana) era la señal clave —
91.8% de importancia en la selección de features del EDA.
**Medimos**, con 657 experimentos y revalidación oficial:

| Conjunto | Features | Accuracy |
|---|---:|---:|
| Extendido completo | 49 | 86.29 |
| Extendido **sin lags semanales** | 45 | **86.76** |

**Implica:** quitar cuatro features sube 0.47 puntos. La importancia que
reporta un Random Forest mide cuánto *usa* una señal, no cuánto *ayuda*.

Los dos resultados conviven y ahí está lo interesante: Random Forest **sí**
pierde 2.11 puntos sin `lag_672`, y CatBoost **gana** 0.47. No es una
propiedad del dato sino de cada modelo. Lo que llamábamos "brecha de
fragilidad ante drift" era el costo de sobreajustarse a esa señal.

**Confirmado en el pipeline:** el candidato compitió contra los otros 12 con
el protocolo oficial y quedó como champion (86.76 contra 86.61). El riesgo
que el informe señalaba como principal —que `lag_672` dejara de ser fiable—
dejó de aplicar, porque el champion ya no la usa.

### 4. Especializar por horizonte empeora
**Creíamos:** un modelo por horizonte (+15, +30, +45, +60) sería mejor que
uno solo.
**Medimos:** 86.39 especializado contra 86.93 compartido.
**Implica:** el modelo único gana porque aprende de cuatro veces más datos.
La intuición de "especializar es mejor" falla cuando parte los datos.

### 5. La identidad de la estación era una señal ausente
**Medimos:** agregarla como one-hot dio +0.94, más que cualquier ajuste de
hiperparámetros.
**Implica:** el EDA ya mostraba 3x de diferencia en demanda media entre
estaciones, pero ninguna feature se lo decía al modelo explícitamente.

### 6. Usábamos el clima equivocado
**Medimos:** la API entrega `rain_forecast` y `temperature_forecast`, y
nunca se usaron. Se alimentaba el clima *observado* para predecir el futuro.
**Implica:** para predecir, el pronóstico es la variable correcta; lo
observado describe el pasado.

### 7. Alinear el objetivo con la métrica no sirvió
**Creíamos:** entrenar con error cuadrático mientras nos califican con WAPE
era un desajuste que costaba puntos.
**Medimos:** objetivo MAE 86.56, pesos por estación 86.63, actual 86.59.
**Implica:** diferencias dentro del ruido. La hipótesis era razonable y
resultó falsa; se descartó en vez de forzarla.

### 8. El ensamble aporta poco y cuesta mucho
**Medimos:** ensamble 86.61 contra 86.47 del mejor individual: +0.14. Pero
pesa 38 MB y necesita tres librerías en producción.
**Implica:** sin caché serían ~6 GB de descarga por semana de competencia,
por encima del plan gratuito. La ganancia no justificaba el riesgo
operativo.

**Desenlace:** la disyuntiva se disolvió sola. El CatBoost del hallazgo 3
resultó **más preciso y más simple** a la vez (86.76, 19 MB, una
dependencia), así que no hubo que elegir entre precisión y operación. Vale
anotar que fue suerte, no diseño: la decisión estaba tomada para sacrificar
0.14 puntos si hacía falta.

### 9. Estamos cerca del techo
**Medimos:** un modelo perfecto que conociera la media real de cada franja
horaria alcanzaría 88–89. Nosotros vamos en 86.76.
**Implica:** quedan ~2 puntos de margen teórico. Perseguir el 90 sobre este
histórico no es realista.

---

## Sobre los datos

### 10. El techo no baja de noche
**Creíamos:** de madrugada el modelo rinde peor.
**Medimos:** el techo teórico es plano (~88 a las 3am y a las 3pm), y el
efecto real de la hora sobre nuestro modelo es de **1 punto**, no de trece.
**Implica:** ver un ciclo nocturno malo no es señal de nada.

### 11. Un ciclo suelto no dice nada
**Medimos** sobre 2.229 ciclos evaluados fuera de muestra:

| | |
|---|---:|
| Accuracy media por ciclo | 85.74 |
| Desviación | 2.60 |
| Peor ciclo | 56.43 |
| Percentil 5 | ~79 |

**Implica:** cada ciclo se evalúa con cuatro puntos por estación. Un 79 es
normal; el mínimo de 56 ocurrió sin que nada fallara.

---

## Sobre la medición

Los hallazgos más útiles del proyecto no fueron sobre el modelo, sino sobre
cómo medirlo.

### 12. Cinco mediciones no son una conclusión
**Creíamos**, tras un simulacro: el accuracy caía 13 puntos de madrugada.
**Medimos** con 2.229 ciclos: era 1 punto. Los 72.17 que dispararon la
alarma eran el 0.6% peor, por azar.
**Implica:** se sacó una conclusión de cinco puntos y era falsa. El propio
error se corrigió midiendo más.

### 13. Dos accuracies con el mismo nombre no son comparables
**Medimos:** la validación agregada da 86.61; el promedio de ciclos
individuales da 85.74. Son formas distintas de agregar lo mismo.
**Implica:** comparar una contra otra metía 0.87 puntos de sesgo en el
detector de drift. Cualquier comparación exige el mismo protocolo.

### 14. La exploración barata sí sirve para preseleccionar
**Medimos:** la diferencia media entre validar con 3 cortes y con 5 fue de
**0.11 puntos**.
**Implica:** explorar barato y revalidar caro es una estrategia válida —
pero los números de exploración no se pueden comparar contra el champion.

### 15. Ventanas solapadas subestiman la variabilidad
**Medimos:** ventanas corridas hora a hora comparten 23 de sus 24 ciclos, y
su dispersión daba 0.19 cuando la real es 1.51.
**Implica:** el umbral quedaba pegado a la base y producía falsas alarmas.
Hay que estimar la desviación con bloques independientes.

### 16. Contar episodios, no chequeos
**Medimos:** 101 alarmas en una serie de 43 días parecían inaceptables.
Resultaron ser **3 episodios**: una misma excursión dispara muchos chequeos
seguidos.
**Implica:** elegir la métrica equivocada casi nos hace romper un detector
que estaba bien. Lo que importa es cada cuánto alguien tiene que ir a mirar.

### 17. Un detector puede taparse a sí mismo
**Medimos:** una caída de 10 puntos se detectaba el 2% de las veces, menos
que una de 6 (10%).
**Implica:** que a mayor degradación hubiera menos detección delataba el
error — el período de referencia incluía ventanas ya contaminadas por la
caída. Un resultado no monótono casi siempre es un bug, no un hallazgo.

### 18. Una etiqueta equivocada sobrevive más que un bug
**Creíamos:** el entrenamiento imprimía "Brecha de fragilidad: 2.11 puntos" y
el informe lo llamaba el riesgo principal del proyecto.
**Medimos:** el número era correcto; la frase, no. Medía cuánto se apoya
*Random Forest* en una señal, no cuánto costaría perderla.
**Implica:** nada falla cuando el nombre está mal. El valor se imprimió
correctamente durante días mientras el texto que lo acompañaba afirmaba lo
contrario de lo que el dato sostenía. Un bug se delata; una interpretación
equivocada hay que ir a buscarla.

### 19. Un test que pasa no prueba nada
**Medimos:** se rompió el collector a propósito de tres formas y se verificó
que cada una hiciera fallar un test.
**Implica:** sin esa comprobación, un test podría estar pasando también con
el código roto. Seis de los ocho bugs del proyecto pasaban todos los
controles en verde.

---

## Sobre entrenar y predecir

### 20. Dos caminos que deberían ver lo mismo, y no lo veían
**Creíamos:** si el entrenamiento valida en 86.76 y la inferencia entrega
48 predicciones aceptadas, el sistema funciona.
**Medimos:** la primera entrega oficial sacó **20.37**. La inferencia
mandaba en 0 las cinco features de hora del objetivo (solo existían en el
camino de entrenamiento), no pedía el pronóstico del clima, y leía la hora
en UTC cuando el modelo había aprendido la de Bogotá. Corregido, el mismo
ciclo da **82.86**.
**Implica:** entrenar bien y predecir bien por separado no basta; hay que
comprobar que los dos caminos construyan el mismo vector de features. Toda
feature debe tener una sola definición, compartida, y la inferencia debe
fallar ante una columna que no sabe calcular en vez de rellenarla con 0.

### 21. El mismo instante no es la misma hora
**Medimos:** el CSV trae UTC-05:00 y Supabase devuelve UTC. Lags,
diferencias y horizontes salen idénticos por los dos caminos, así que
ninguna comparación de valores lo detectaba. Solo `.dt.hour` cambia: 7 por
un lado, 12 por el otro.
**Implica:** normalizar la zona horaria en un único punto, al construir las
features, y probarlo con los mismos datos expresados en las dos zonas.

---

## Sobre la infraestructura

### 22. GitHub descarta corridas programadas
**Medimos:** 2 ejecuciones de ~28 esperadas en 7 horas. No era cuota ni
configuración: bajo carga, GitHub no encola las atrasadas, las descarta.
`:00` y `:30` son los minutos más congestionados.
**Implica:** perder una corrida en competencia cuesta una ventana de 25
minutos. Se mitiga con más intentos en minutos impares y vigilancia interna
por corrida.

### 23. PostgREST nunca devuelve más de 1000 filas
**Medimos:** pedir `limit=5000` devuelve 1000, con
`Content-Range: 0-999/51840`.
**Implica:** una paginación que asuma páginas más grandes corta en la
primera. Nos cargaba 1 de 12 estaciones, y solo apareció con volumen real.

### 24. Otras rarezas del entorno

| Qué | Detalle |
|---|---|
| Supabase y PowerShell | `Invoke-RestMethod` con la clave secreta da 403: lo toma por navegador. Usar `curl`. |
| SQLite sobre rutas de red | No puede tomar bloqueos: "database is locked". La base de MLflow va a una carpeta local. |
| MLflow 3.x | Retiró el backend de archivos; exige base de datos. |
| `TaskStop` | No mata el proceso hijo. Un barrido siguió vivo ocupando 13 GB y hacía morir todo lo demás **sin dar ningún error**. |

---

## El patrón

Seis de los ocho bugs no lanzaban ninguna excepción. Los tests pasaban. El
código se veía bien. Aparecieron al **ejercitar el sistema con datos reales y
a escala**, no al probar casos sueltos.

Y tres de ellos se encontraron buscando otra cosa: el simulacro que se hizo
para validar el pipeline de evaluación destapó el problema del monitoreo; la
batería de esfuerzo, escrita para medir sensibilidad, destapó que el
detector se anulaba solo.

Los dos últimos los encontró la competencia misma: la API aceptó la entrega
y solo el accuracy del leaderboard delató el problema. Es el argumento más
fuerte para medir con datos reales cuanto antes.

---

## 25. Un modelo congelado no puede ganarle al drift (y un perfil sí)

El champion se estancó en ~83 de accuracy justo cuando el generador aplicó
`peak_shift` sobre cuatro estaciones (02300, 06111, 07107, 09122: el pico se
corre 45 min y el nivel sube 18%). Pasaron de ~85 a ~63-70 y sobrepredecían
entre 14% y 31%.

Antes de aceptar la explicación se descartaron tres hipótesis, y vale la pena
anotar **lo que no era**, porque cada una parecía razonable:

| Hipótesis | Prueba | Resultado |
|---|---|---|
| El contexto (lluvia/temperatura) se cortó el 9-sep y llega como NaN | Predecir con contexto real, imputado por hora, último visto y cero | **83.6 en las cuatro**: el modelo casi no usa esas variables |
| Falta reentrenar con datos recientes | Reentrenar hasta el 11-sep, sin pesos y con vida media 14 y 5 días | **82.2-82.7**: *peor* que el champion. Con 1.5 días post-drift se añade ruido, no señal |
| Corregir el sesgo reciente por estación | Factor `real/predicho` de las últimas 6-48 h, con y sin encogimiento | **+0.3 a +0.7**: dentro del ruido |

Lo que sí funcionó fue cambiar de familia. El nombre del modelo de un
competidor que se recuperó rápido del drift (`adaptive-profile-hl14`, visible
en el portal) apuntaba a un perfil adaptativo, y replicarlo lo confirmó:

    predicción = perfil(estación, franja, tipo de día)  x  factor_de_nivel(última hora)

El perfil es una media de la misma franja en días previos del mismo tipo con
vida media de 14 días; el factor compara la última hora real contra lo que el
perfil esperaba. **No se entrena nada**: el factor absorbe un cambio de nivel
en la hora siguiente, no en el próximo reentrenamiento.

Validado con origen rodante sobre 5 ventanas de un día (nunca sobre una sola,
justamente para no confundir suerte con mejora):

| Modelo | 7-sep | 8-sep | 9-sep | 10-sep | 11-sep (drift) | Media | Peor |
|---|---|---|---|---|---|---|---|
| Champion CatBoost | 90.92 | 91.35 | 85.27 | 85.37 | **83.41** | 87.27 | 83.41 |
| Perfil solo (hl=14, L=4) | 87.07 | 87.87 | 86.42 | 86.24 | 85.53 | 86.62 | 85.53 |
| **Mezcla 60/40** | 89.84 | 90.36 | 86.48 | 86.67 | **85.29** | **87.73** | **85.29** |

La mezcla **cede ~1 punto en los días tranquilos y gana ~1.9 en el día con
drift**. Se promovió por el peor caso, no por la media: la tabla se decide en
ventanas de seis ciclos, así que un desplome de un día pesa más que una
décima de promedio. El profesor ya anunció un segundo drift (`closure`: una
estación cae al 40% y reparte a las vecinas), y ahí el perfil vuelve a ser
quien reacciona primero.

Dos detalles de la implementación que no son opcionales:

- El factor de nivel se **recorta a [0.5, 1.6]**. Sin recorte, un hueco del
  collector se convierte en un multiplicador absurdo.
- Si el perfil no tiene opinión (estación nueva, franja sin historia) devuelve
  `None` y la entrega sale con el champion solo. **Perder un ciclo cuesta
  mucho más que entregarlo un punto peor.**

---

## 26. El profesor fijo el Corte 1 y no definio aun su cierre

El `contract-watch` se puso en rojo el 28-sep: seis commits nuevos del profesor.
Ninguno toca el camino de entrega (submissions, ciclos y campos siguen
identicos; `check_contract.py` lo confirma contra la API real), pero dos de
ellos cambian las reglas de evaluacion y valia leerlos:

- **El Corte 1 arranca el 25-sep a las 00:00 Bogota (`2026-09-25T05:00:00Z`)**,
  con inicio fijo, no ventana movil. Los ciclos del 21 al 24 quedan como
  historico de aprendizaje y **no cuentan**: ni en el acumulado, ni en la
  ventana de seis ciclos, ni en los totales de entregas.
- **El fin del corte todavia no esta fijado.** El documento
  `docs/primer-corte-evaluacion.md` dice explicitamente que antes de cerrar una
  nota hay que fijar ese fin y guardar un snapshot reproducible. Nuestra
  estimacion de "termina el 28-sep" salia de `competition_days: 7` en
  `config/scenario.example.yaml`, que es un EJEMPLO: no hay fecha confirmada, y
  conviene seguir entregando hasta que el reloj deje de correr.
- La fase de drift empezo "la noche del 25 de septiembre", que es exactamente
  cuando vimos el `peak_shift` y despues el `closure`.
- Un target no entregado se evalua como prediccion cero, y los reintentos del
  mismo ciclo no suman ciclos: cuenta el intento oficial.

El leaderboard por API ya refleja este corte: `cumulative` trae `starts_at` y
`resolved_cycles` desde el 25-sep, y `accuracy_at_20` viene en `null` en esa
ventana a proposito.

La leccion operativa: la alarma de contrato sirvio para enterarnos de un cambio
de REGLAS, no de un cambio de API. Vale la pena leer los commits del profesor
aunque el pipeline siga funcionando.

## 27. La continuacion del escenario supero los topes del factor de nivel

El 28-sep a las ~15:50 UTC el reloj paso a `state: "waiting"` durante ~5.7 h
(sin ciclo abierto, sin penalizacion: confirmado en `docs/fase-drift.md`, "el
escenario original agoto su horizonte"). Un PR del profesor (`julianzu9612`,
28-sep) implemento una continuacion versionada con "dificultad" 0-3 y fijo el
**cierre real: viernes 2 de octubre 2026 a las 23:59 Bogota**.

Tras la reactivacion, el accuracy por ciclo (no el promedio movil 24h, que
suaviza esto) cayo de 92.87 a 75.74 en 5 horas (ciclos T100000Z-T150000Z),
con sesgo fuertemente negativo. El diagnostico en vivo mostro el factor de
nivel pegado a los dos topes a la vez, en estaciones distintas:

- **02300 y 05000**: `factor_de_nivel` pegado al tope superior (1.6) en las
  ventanas corta Y larga durante 4 ciclos seguidos. pred/real = 0.44-0.58: la
  demanda real subio a ~2.0x y el tope no dejaba seguirla.
- **05100**: cierre detectado (`peso_perfil=1.0`), factor pegado al piso
  (0.30), pero pred/real SUBIO 1.11 -> 1.57 ciclo a ciclo: la demanda real
  seguia cayendo por debajo del 30% que el piso permitia representar.

Backtest causal (mismas observaciones que se habrian tenido en cada ciclo,
comparado contra el ground truth ya resuelto):

| estacion | accuracy con tope viejo (0.30, 1.6) | con tope nuevo (0.15, 2.5) |
|---|---|---|
| 02300 | 58.99 | 74.62 |
| 05000 | 52.69 | 77.38 |
| 05100 | 73.46 | 79.42 |

Chequeo de regresion sobre 8 ciclos sanos previos (384 predicciones, 12
estaciones): **0 predicciones cambiaron**, WAPE identico al decimal. El
tope solo se activa cuando la demanda real ya se desvio >60% del perfil, asi
que ampliarlo no cuesta nada en regimen normal — mismo razonamiento que bajar
el piso de 0.5 a 0.3 la primera vez (hallazgo original en `perfil.py`).

`LIMITES_FACTOR` paso de `(0.30, 1.6)` a `(0.15, 2.5)` (commit `8c1a29d`).
La leccion: el promedio movil de 24h que usa el vigilante de drift esconde
caidas reales de 3-5 horas en estaciones puntuales porque las diluye entre 24
ciclos buenos. Hay que revisar tambien el accuracy POR CICLO (el dashboard lo
grafica) y no solo el promedio movil.

## 28. El profesor cambio la FORMA de la demanda (no solo el nivel) y faltaba
    el detector simetrico de alza

El 30-sep a las 14:52 hora Bogota el profesor activo la "revision 2" de drift
(nivel 3) y poco despues la "revision 3", mas exigente aun (`docs/drift-
operations.md` del repo del profesor). Su propia bitacora es explicita:

> "La continuacion privada modifica la forma temporal de la demanda... La
> antigua referencia adaptativa que solo ajustaba la escala del perfil deja
> de ser apropiada para este cambio de forma."

Calibracion propia del profesor para la revision 3 (referencias privadas,
no nuestras notas): fija con variables recientes 55.0-55.6%, la MISMA
familia reentrenada periodicamente 79.4-80.3% (recupera a ~84% entre las
horas 18-30; las primeras 6 horas son duras incluso para el modelo
adaptativo, ~55-56%).

El accuracy por ciclo confirmo el golpe: de 89-91% cayo a 67.4% en una hora
(ciclo T040000Z del 18-sep). El diagnostico en vivo mostro la mitad del
mecanismo que ya teniamos (el `LIMITES_FACTOR` ampliado del hallazgo #27) y
la mitad que faltaba: **02300 y 05000 llevaban horas con el factor de nivel
pegado al tope (2.5) en las DOS ventanas, pero al no tener desfase se
quedaban en `MEZCLA_PERFIL=0.4` de siempre** - el champion, con 60% de la
mezcla y ciego al evento, arrastraba la prediccion hacia abajo. `06000` paso
de factor4=1.28 a 2.50 en una sola hora (entre los anclas de los ciclos
T040000Z y T050000Z) y su accuracy se desplomo de 82 a 37 en el mismo salto.

El cierre ya tenia su detector (hallazgo original): cuando el factor se
desploma en las dos ventanas, se le quita la voz al champion. Faltaba el
espejo: cuando el factor se DISPARA en las dos ventanas, pasa exactamente lo
mismo pero al reves, y no habia nada que lo capturara salvo el desfase (que
solo cubre corrimientos de fase, no saltos de nivel puros).

Se agrego `UMBRAL_ALZA_CORTO=1.30`, `UMBRAL_ALZA_LARGO=1.20`,
`PESO_PERFIL_ALZA=1.0`, chequeado en `peso_de_mezcla` justo despues del
cierre. Validacion:

- Con el perfil solo como referencia optimista sobre las 7 horas del
  episodio (17-sep 21:00 a 18-sep 04:00): 02300 87.3%, 05000 87.7%,
  05100 89.4%, 03000 86.6% - muy por encima de lo que puede dar un champion
  ciego al evento.
- 180 combinaciones estacion x hora de regimen sano (5 al 9 de septiembre,
  antes de cualquier drift): **cero disparos falsos**.
- Test unitario `test_una_alza_sostenida_le_quita_la_voz_al_champion`
  (simetrico a `test_un_cierre_le_quita_la_voz_al_champion`). Nota interna:
  el detector de fase (`desfase`) mostro una asimetria en el escenario
  sintetico de alza sostenida por mas de 4 horas (encuentra un desfase
  espureo que no aparece en el escenario simetrico de cierre); no se
  investigo a fondo por la urgencia, pero no bloquea el caso real porque el
  peor resultado posible es caer a `PESO_PERFIL_CON_DESFASE=0.70` en vez de
  `1.0` - sigue siendo mejor que el `0.4` de antes. Vale la pena revisarlo
  con mas tiempo.

Commit `ee8675c`. El cierre real de la competencia sigue fijo el viernes 2 de
octubre 23:59 hora Bogota (confirmado otra vez en la misma bitacora del
profesor).

## 29. La revision 3 de drift supero el tope de 2.5 en menos de un dia

El detector de alza (#28) funciono como se diseño - peso=1.0 activo en casi
todas las estaciones a la primera senal - pero el tope de 2.5 resulto
insuficiente para la magnitud real de la revision 3. Accuracy por ciclo:
~90 (17-sep 23:00) -> 67.4 -> 37.4 -> 34.6 en cuestion de horas, con las 12
estaciones cayendo simultaneamente (antes eran 2-4 estaciones puntuales).

Medido sin tope en vivo (ancla 18-sep 06:00): 07111 en 6.42x, 02300 en 5.95x,
la mayoria entre 3x y 6x, con el factor CORTO por encima del LARGO en casi
todas las estaciones - la subida seguia acelerando, sin estabilizarse
todavia. Backtest causal contra el ciclo T050000Z ya resuelto (ground truth
real): tope 2.5 -> 41.3 de accuracy (perfil solo); tope 5.0 -> 50.6; 8.0 y
12.0 no mejoran mas alla de 5.0 para ESE ciclo puntual, pero la demanda
seguia subiendo despues de resuelto. Se subio el tope a 10.0 (commit
`84c1390`), con margen sobre el 6.42x ya visto, para absorber una subida
que aun no toco techo sin tener que repetir el ajuste en un par de ciclos.
Chequeo de regresion: 0 predicciones cambiaron en 15 anclas de regimen sano.

Leccion: un tope que se subio de forma justificada y validada (#27) igual
puede quedar corto si el evento real resulta mas severo que la evidencia
disponible en el momento de fijarlo. La senal para revisarlo de nuevo es la
misma: el factor pegado EXACTO al tope en la mayoria de las estaciones a la
vez, con el accuracy por ciclo (no el promedio movil de 24h) cayendo fuerte.

---

## 30. Subir el tope de nuevo NO habria arreglado la caida de T050000Z/T060000Z

**Que creiamos:** con 09122 y 07105 pegadas exacto al tope de 10.0 en vivo
(factor4 sin tope = 10.40 y 10.28) y el accuracy por ciclo desplomado a 37.4
(T050000Z) y 48.8 (T060000Z) -muy por debajo del umbral de reentreno de
85-, la hipotesis obvia era: el tope de nuevo se quedo corto, subirlo otra
vez.

**Que medimos:** se corrio exactamente el mismo backtest causal usado en
#27/#29 pero contra el ANCLA REAL del ciclo T060000Z ya resuelto (06:00Z,
no el instante actual). El factor sin tope en ESE ancla fue 2.34-3.30 para
las tres estaciones peor libradas (09122/07105/05000) - muy por debajo
incluso del tope viejo de 2.5, nunca cerca del tope vigente de 10.0. Se
confirma reconstruyendo las predicciones completas con tope 10, 15, 20 y 30:
el batch de predicciones sale BYTE IDENTICO en los cuatro casos (357.6,
363.7, 352.3, 360.5 para 09122, etc.) mientras la demanda real resulto 2.2 a
3.5 veces mas alta (805-1627 contra 352-481 predicho). Subir el tope no
cambia nada porque el tope nunca se activo en ese ancla: el perfil ya tenia
peso 1.0 (ALZA detectada, funcionando) y aun asi se quedo corto.

Por separado, el chequeo en vivo (ancla 07:30Z, una hora despues) si muestra
el factor pegado al tope para 09122/07105 (10.40/10.28), pero por apenas
2-4% - nada parecido al 6.42x que motivo subir a 10.0 en #29.

**Que implica:** la caida de estos dos ciclos NO fue un problema de tope.
Fue que la demanda crecio mas rapido de lo que el perfil (ya con peso
completo) puede seguir dentro de una sola hora - exactamente lo que el
profesor describio como cambio de FORMA, no de escala (#28), y coincide con
su propio dato de calibracion: incluso un modelo adaptativo que se
reentrena solo saca 55-56% en las primeras horas de una revision de drift
nueva, recuperando hacia 84% entre las horas 18-30. Es un limite real del
mecanismo (correccion de escala sobre un patron recency-weighted), no un
bug ni un parametro mal puesto. Evidencia de que ya esta sanando: de 12
estaciones, 9 mejoraron su accuracy entre T050000Z y T060000Z (03000
22.8->84.0, 09000 37.2->79.8, 10009 0.0->49.0, etc.); solo 09122/07105/05000
siguieron empeorando ciclo a ciclo, el mismo trio que esta en la cola de la
"ola" ya documentada en #29. Decision: NO se toca `LIMITES_FACTOR` de
nuevo con esta evidencia - subirlo mas no habria cambiado el resultado y el
riesgo de sobre-corregir en regimen sano es real. Se sigue vigilando con el
mismo criterio de reapertura de #29.

---

## 31. Una subida se puede revertir de golpe - darle peso 1.0 al perfil en ALZA sale caro

**Que creiamos:** el peso completo (1.0) para el perfil durante una alza
sostenida (#28) era simetrico al de un cierre y seguro por la misma razon:
"el champion esta ciego al evento, confiar en el perfil es mejor".

**Que medimos:** el ciclo `T080000Z` (01-oct) resolvio con 05000, 07105 y
09122 en 0.0 de accuracy exacto - las tres en ALZA (peso=1.0). Revisando la
demanda real: las tres venian bajando SUAVE hasta el ancla de prediccion
(08:00) y luego se desplomaron de golpe en la hora siguiente (05000:
1627->1616->1456->1310->1047->910 y DESPUES 910->534->393->225->167; 07105
868->...->418->...->75; 09122 1244->...->541->...->114). El perfil, con peso
completo, extrapolo el nivel alto (876-1252) justo en la hora en que la
demanda real ya se habia hundido. No es un problema de deteccion tardia: al
momento de predecir la caida todavia no habia pasado, no habia ninguna senal
que la anticipara - es information del futuro. El champion solo (204-421)
hubiera quedado mucho mas cerca de lo real (534-167) que el perfil.

Backtest causal con los 4 ciclos del colapso completo (T050000Z a T080000Z,
12 estaciones, 48 predicciones por corrida, ground truth real):

| PESO_PERFIL_ALZA | accuracy agregado |
|---|---|
| 1.0 (el que estaba) | 32.3 |
| 0.7 | 35.1 |
| **0.5** | **35.9 (mejor)** |
| 0.3 | 35.0 |
| 0.0 (solo champion) | 29.8 |

El perfil sigue aportando (0.0 es el peor de todos, confirma que #28 valia la
pena), pero darle el 100% de la voz es peor que una mezcla. Se bajo
`PESO_PERFIL_ALZA` de 1.0 a 0.5 (commit siguiente). Los 63 tests existentes
siguen pasando (ninguno fija el valor numerico, solo comparan contra la
constante).

**Que implica:** una subida sostenida y un cierre NO son igual de seguros de
cara al futuro. Un cierre tiende a persistir una vez empieza; una subida de
esta revision de drift puede revertirse sin aviso en la hora siguiente -
exactamente la clase de "cambio de forma" que el profesor advirtio. La
leccion para el proximo ajuste de este tipo: nunca asumir que dos eventos
"simetricos" en como se detectan son simetricos en que tan bien se puede
confiar en que continuen.

---

## 32. Un detector de "frenada aguda" (ultima lectura vs unos pasos atras) no ayuda - descartado

**Que creiamos:** tras #31, el ciclo `T100000Z` resolvio con OTRAS cinco
estaciones (02300/03000/06000/07111/09000) en 0.0 de accuracy exacto, mismo
patron de pico-y-colapso. En 3 de las 5 la caida YA era visible en las 2-3
lecturas previas al ancla de prediccion (ej. 02300: 1068->779->720, ya
bajando antes de predecir) pero el factor de nivel (promedio de ventana) no
lo reflejaba todavia - la hipotesis era agregar un chequeo rapido sobre la
serie cruda (ultima lectura vs 1-2 pasos atras) para bajarle la voz al
perfil apenas se vea una frenada, sin esperar a que la ventana completa lo
note.

**Que medimos:** implementado y probado contra los 5 ciclos resueltos del
colapso (T050000Z-T100000Z, 12 estaciones, ground truth real), variando el
umbral (0.50-0.60) y la ventana de comparacion (1-2 pasos = 15-30 min):
accuracy agregado 27.78 (el 0.5 plano de #31, la base) contra 27.58-27.87
en las variantes - diferencias dentro del ruido, ninguna mejora real.
Aparentemente el detector dispara tanto en frenadas genuinas como en el
vaiven normal de una subida real (que tambien tiene bajones de una lectura
sin ser el pico), y lo que gana en unos casos lo pierde en otros.

**Que implica:** no se aplica - se queda la correccion de #31 (peso 0.5
plano) sin este refinamiento. Queda descartada esta forma puntual del
chequeo para no reinvestigarla; una version mas elaborada (ej. pendiente
sobre 3+ puntos en vez de un cociente de 2, o normalizada por la volatilidad
tipica de cada estacion) podria funcionar distinto, pero no se probo por
tiempo. El colapso tan rapido y profundo (miles a cientos en una hora) hace
que CUALQUIER modelo anclado en el nivel reciente sobrestime fuerte en
terminos relativos (WAPE se dispara contra un denominador chico) - puede que
el limite real aqui no sea de deteccion sino de lo que es forecasteable con
1 hora de anticipacion en este regimen.

---

## 33. El enfoque de otra estudiante (autorizado por el profesor) - revisado, no adoptado

**Contexto:** Kevin le pregunto directamente al profesor si podiamos revisar
el enfoque de una companera (Maria Isabell Guzman Faneyte, `extra_trees_
regressor_v1`, 58% en el leaderboard del momento) para mejorar el nuestro.
El profesor autorizo explicitamente por correo, y ella confirmo que no le
molestaba. Con permiso confirmado de ambas partes, se ubico su fork publico
(`alomariaDev/pulso-transmi-sdk`, identidad confirmada por el email de sus
commits) y se reviso su codigo de inferencia.

**Que hace distinto:** arquitectura fundamentalmente distinta a la nuestra.
Nosotros: champion fijo (entrenado una vez, solo se reemplaza si un
reentreno le gana en CV) + perfil adaptativo como capa de correccion aparte.
Ella: **reentrena un ExtraTreesRegressor completo en CADA ciclo** (cada 10
min via GitHub Actions) con los datos mas frescos disponibles, SIN perfil
adaptativo separado - y predice los 4 horizontes de forma RECURSIVA (predice
+15min, usa esa prediccion como si fuera una observacion real para calcular
los lags de +30min, y asi sucesivamente), en vez de nuestro enfoque directo
(un horizon_minutes como feature, prediccion independiente por horizonte
desde la misma ancla). Al reentrenar tan seguido, su modelo base siempre
"conoce" las ultimas horas de drift sin depender de una capa de correccion
aparte. Tambien corre un detector de drift por PSI (Population Stability
Index, ventana de 7 dias) para decidir cuando vale la pena reentrenar, mucho
mas lento que nuestro mecanismo (chequeo cada 30 min contra el umbral fijo
de 85% que pidio el profesor).

**Que medimos:** se replico su pipeline exacto (mismas features: lags de 15
min/1h/1d/7d, rolling mean/std, hora/dia, prediccion recursiva) entrenado
SOLO con los ultimos 7 dias de nuestros propios datos (igual que ella),
contra los dos ciclos del colapso ya analizados (#31, #32), comparando
contra nuestro accuracy real de produccion en esos mismos ciclos:

| Ciclo | Nuestro accuracy (produccion real) | Enfoque de ella (replicado con nuestros datos) |
|---|---|---|
| T080000Z | 43.5 | 42.5 |
| T100000Z | 32.7 | 35.7 |

Prácticamente empatado - mejor en un ciclo, peor en el otro, diferencias
chicas frente al ruido normal entre ciclos.

**Que implica:** no se adopta esta arquitectura. No es que la idea sea mala
-el reentreno frecuente es una estrategia legitima contra drift, y la
prediccion recursiva tiene merito- pero con solo 2 ciclos de evidencia y una
señal mixta (no una ventaja clara y consistente), reescribir el pipeline de
produccion (que corre cada hora, en vivo, sin margen de error) seria
exactamente el tipo de cambio no validado que esta disciplina busca evitar.
Si en el futuro hay tiempo para un backtest mas amplio (10+ ciclos, variando
el tamano de la ventana de entrenamiento) y la señal se sostiene, vale la
pena reconsiderarlo - probablemente como una tercera voz en la mezcla
(champion + perfil + un modelo liviano reentrenado por ciclo) en vez de un
reemplazo total.

---

## 34. Extrapolacion lineal pura en el caso EXTREMO - +20 a +23 puntos en el colapso

**Que creiamos:** tras #31 (bajar `PESO_PERFIL_ALZA` a 0.5) y #32/#33 (nada
mas probado ayudaba), el siguiente paso logico era buscar en la literatura de
forecasting bajo concept drift. La investigacion confirma el principio: el
aprendizaje online/incremental recupera de un drift en menos de 1 hora,
contra 24h-7 dias de un reentreno por lotes (fuentes: [Online Learning vs
Batch Retraining](https://inferensys.com/differences/logistics-and-supply-chain-visibility-ai/ai-driven-demand-forecasting-models/online-learning-vs-batch-retraining-for-demand-shift-adaptation),
[When to Retrain (arXiv 2608.19488)](https://arxiv.org/pdf/2608.19488),
[Proactive Model Adaptation Against Concept Drift (arXiv 2412.08435)](https://arxiv.org/html/2412.08435v3)).
La version mas simple y barata de "online" es extrapolar linealmente la
serie cruda reciente, sin pasar por ningun patron historico.

**Que medimos:** probado con el protocolo de siempre (backtest causal, ground
truth real, metrica oficial - accuracy por estacion promediado, NO WAPE
global pooled, que es una metrica distinta y mas optimista que se uso por
error en una primera pasada de este mismo hallazgo):

- Extrapolacion lineal pura (ultimas 4 lecturas, 1h) contra los dos ciclos
  del colapso (T080000Z, T100000Z): 75-77 de accuracy, muy por encima de
  cualquier cosa probada hasta ahora.
- Reemplazando SOLO el perfil historico en las estaciones que ya estaban en
  alza/cierre (dejando el resto de la mezcla igual): produccion real 41.23 y
  32.70 -> con extrapolacion 64.35 y 52.12. **+23 y +19 puntos.**
- Chequeo de regresion en 8 ciclos de regimen sano (incluye alza/cierre
  moderados ya cubiertos por los umbrales normales): activar la
  extrapolacion en CUALQUIER alza/cierre (umbral normal) cuesta -1.87 puntos
  de promedio (81.65 vs 83.52) - el perfil historico SI aporta forma en
  casos moderados que una recta no tiene.
- Barrido de un umbral mas exigente (solo activar si el factor esta MUY lejos
  de 1.0, no en el borde): 2.0 y 2.5 conservan el beneficio completo del
  colapso (64.35/52.12 identico) con -1.16 y -0.82 de costo en regimen sano;
  3.0 ya empieza a perder beneficio en el colapso (49.32/46.12). Se eligio
  2.5/0.4 (alza/cierre) como el mejor punto medido.

**Que implica:** se agrego `extrapolacion_extrema()` a `PerfilAdaptativo`
(`src/perfil.py`) y se conecto en `infer.py` ANTES de la mezcla
champion+perfil de siempre: si el factor de nivel esta en zona extrema
(>2.5 o <0.4), la extrapolacion lineal manda sola; si no, todo sigue igual
que antes (#28/#31). Es la mejora de accuracy mas grande medida en toda esta
investigacion del drift. 4 tests nuevos (67/67 en total). La leccion de #31
se confirma y se agudiza: no solo "cuanto peso darle" al perfil importa, sino
que el perfil historico en si mismo deja de ser la herramienta correcta
cuando el evento es tan extremo que ningun dia anterior se le parece - ahi
lo que sirve es la tendencia de la ULTIMA hora, no un patron de semanas.

**Seguimiento en vivo (mismo dia):** el primer ciclo real con esto activo
(`T190000Z`) disparo la extrapolacion en 6 estaciones. Resultado mixto pero
revelador: 07107 y 10009 se recuperaron a 90.3 y 91.5 (la extrapolacion
funciono exactamente como se diseño); 05000/07105/09122 siguieron mal (5.1 a
30.1) porque el colapso tambien revirtio DENTRO de la ventana extrapolada, un
limite inherente de proyectar una recta a ciegas. Ademas aparecieron 06111 y
09000 en 0.0 exacto SIN disparar el mecanismo: su caida fue mas GRADUAL (204
-> 40 en 45 min, no un salto instantaneo) y el factor nunca cruzo el umbral
extremo de 2.5/0.4, quedandose con la mezcla normal que tampoco acerto. La
"ola" sigue moviendose y presentando variantes (colapso instantaneo vs
gradual) que este primer corte no cubre completo. No se parcha mas esta
noche sin validar primero - queda como evidencia para la proxima revision
con la cabeza fresca, no para una reaccion apurada.

**Segundo seguimiento (T050000Z, 21.6 de accuracy, el peor desde el fix):**
cuatro estaciones en 0.0 exacto, cada una por una razon distinta - confirma
que son variantes del mismo limite, no un bug nuevo. 09122 hizo una reversion
en V dentro de la misma hora (880->108->661): la extrapolacion vio la caida
hacia el ancla y proyecto que seguia bajando, dando 0.0 justo cuando rebotaba
con fuerza - el caso mas dificil posible para una recta, un fondo de valle
en vez de una tendencia sostenida. 05100/10009 son el patron gradual ya
documentado (no cruzan el umbral extremo). 07107 disparo la extrapolacion
pero el ruido de las ultimas 4 lecturas le dio pendiente positiva justo
cuando la demanda real ya iba en picada. Ninguno es nuevo ni amerita otro
cambio apurado a esta hora - la extrapolacion sigue siendo una mejora neta
medida (hallazgo #34), simplemente no es magia contra cualquier forma de
reversion.

---

## 35. El drift de la revision 3 es una ONDA de ~4 horas - naive estacional corto: ultimos 12 ciclos 41 -> 92

**Que creiamos:** que los colapsos y rebotes de #31-#34 eran eventos sueltos
(picos que se revierten, reversiones en V) dificiles de anticipar con 1 h de
horizonte, y que la palanca era la capa adaptativa o reentrenar.

**Que medimos:**
- Barrido de los 17 parametros de la capa adaptativa (descenso coordenado
  sobre 161 ciclos, validacion en los 70 mas recientes): la mejor combinacion
  en train EMPEORA en validacion (69.1 -> 67.8). Sobreajuste; la capa no es la
  palanca (el mejor cambio individual da +0.8).
- EDA del 18/19-sep: la curva diaria desaparece y en su lugar hay una onda
  periodica (2 h alta, 2 h baja). Autocorrelacion maxima en el lag de 16 pasos
  (4 h) en 10 de 12 estaciones (r = 0.65-0.88); 02300 y 07107 a 8 h. El naive
  "lo que paso hace 4 h" acierta 91.2% el 19-sep (63% el 18-sep, mientras la
  onda se establecia) y 9% en regimen normal. Ningun lag del champion (15 min,
  1 h, 1 d, 1 sem) puede verla, y el perfil diario tampoco.
- Selector: por estacion y ciclo, con datos <= ancla, el periodo P en [2 h, 6 h]
  que mejor habria acertado en las ultimas 12 h; si su acierto retrospectivo
  >= 80%, prediccion = promedio de target-P y target-2P. Backtest sobre los 231
  ciclos oficiales resueltos, metrica oficial:

| | todo | regimen normal | drift | ultimos 12 ciclos |
|---|---|---|---|---|
| pipeline actual | 80.14 | 85.25 | 59.04 | 41.01 |
| + onda corta | **83.29** | **85.25** | **75.20** | **91.78** |

  Cero regresion en regimen normal (el detector nunca se activa ahi). Periodos
  hasta 12 h o ventanas de 4 h si daban falsas activaciones en normal (-1 a -5
  puntos) - por eso P cabe al menos 2 veces en la ventana. Reproducido
  exacto con el codigo de produccion (no solo con el script del experimento).

**Que implica:** `PerfilAdaptativo.prediccion_periodica()` se consulta ANTES de
la extrapolacion extrema y de la mezcla champion+perfil en `infer.py`. 2 tests
nuevos (69/69). La leccion: ante un drift, mirar la FORMA de la serie (EDA,
autocorrelacion) antes de tunear modelos - toda la noche se ataco el sintoma
(colapsos sueltos) en vez de la estructura (una onda regular).

---

## 36. El contexto se acabo el 8-sep: el reentreno nunca vio el drift y el champion recibia NaN

**Que creiamos:** que cada reentreno automatico aprendia de los datos nuevos.
Daba 86.74 identico corrida tras corrida, y se leia como "el champion sigue
siendo el mejor".

**Que medimos:** la API publica `context` (lluvia, temperatura, pronosticos,
eventos) solo hasta el 2026-09-08 23:45 Bogota; el stream de la competencia
no lo trae y no hay otro endpoint. `train.py` hace `dropna` sobre las
features, asi que toda fila posterior se descartaba: el frame de
entrenamiento terminaba el 8-sep sin importar cuantos datos hubiera (100,032
filas fijas en el experimento de reentreno rodante, en cualquier corte). Y en
produccion, desde el 9-sep el champion recibia las 5 columnas en NaN, algo
que nunca vio al entrenar.

Relleno con la mediana de cada franja de 15 min de las lecturas reales,
recalculando las predicciones del champion en los 231 ciclos: capa completa
83.29 -> 83.52 (normal +0.24, drift +0.16, ningun segmento peor).

**Que implica:** `features.rellenar_contexto()` lo usan inferencia (que ahora
trae TODO el contexto, no solo la ventana de 28 d, para que la mediana no se
quede sin datos en unos dias) y `train.py`. El reentreno ya puede ver el
drift. De paso, la capa adaptativa en `infer.py` quedo envuelta en
try/except por target: un error en UNA estacion entrega esa con el champion
solo, en vez de tumbar las 12.

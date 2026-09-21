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
el código roto. Cuatro de los seis bugs del proyecto pasaban todos los
controles en verde.

---

## Sobre la infraestructura

### 20. GitHub descarta corridas programadas
**Medimos:** 2 ejecuciones de ~28 esperadas en 7 horas. No era cuota ni
configuración: bajo carga, GitHub no encola las atrasadas, las descarta.
`:00` y `:30` son los minutos más congestionados.
**Implica:** perder una corrida en competencia cuesta una ventana de 25
minutos. Se mitiga con más intentos en minutos impares y vigilancia interna
por corrida.

### 21. PostgREST nunca devuelve más de 1000 filas
**Medimos:** pedir `limit=5000` devuelve 1000, con
`Content-Range: 0-999/51840`.
**Implica:** una paginación que asuma páginas más grandes corta en la
primera. Nos cargaba 1 de 12 estaciones, y solo apareció con volumen real.

### 22. Otras rarezas del entorno

| Qué | Detalle |
|---|---|
| Supabase y PowerShell | `Invoke-RestMethod` con la clave secreta da 403: lo toma por navegador. Usar `curl`. |
| SQLite sobre rutas de red | No puede tomar bloqueos: "database is locked". La base de MLflow va a una carpeta local. |
| MLflow 3.x | Retiró el backend de archivos; exige base de datos. |
| `TaskStop` | No mata el proceso hijo. Un barrido siguió vivo ocupando 13 GB y hacía morir todo lo demás **sin dar ningún error**. |

---

## El patrón

Cuatro de los seis bugs no lanzaban ninguna excepción. Los tests pasaban. El
código se veía bien. Aparecieron al **ejercitar el sistema con datos reales y
a escala**, no al probar casos sueltos.

Y tres de ellos se encontraron buscando otra cosa: el simulacro que se hizo
para validar el pipeline de evaluación destapó el problema del monitoreo; la
batería de esfuerzo, escrita para medir sensibilidad, destapó que el
detector se anulaba solo.

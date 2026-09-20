# Dashboard — Pulso TransMi

Página de una sola vista con el estado real del sistema, leída en vivo desde
Supabase. Es el bono *"hacer visible el sistema"* de la guía.

## Qué responde

La guía pide que una buena vista permita siete cosas; seis se resuelven aquí y
una se omite a propósito:

| Lo que pide la guía | Dónde está |
|---|---|
| Reconocer el accuracy actual y su tendencia | Tile "Accuracy real" + gráfica por ciclo |
| Comparar estaciones | Barras por estación, ordenadas de peor a mejor |
| Ver cuál modelo es champion y desde cuándo | Tile "Modelo champion" + tabla de historial |
| Identificar drift sin leer logs | Sección "Señales de drift" |
| Detectar ejecuciones fallidas | Tile "Recolector" + tabla de corridas |
| Relacionar una caída con un cambio de modelo | Marcas verticales de promoción sobre la gráfica |
| Consultar la posición en el leaderboard | **Deliberadamente ausente** (ver abajo) |

## Por qué no muestra el leaderboard

Consultar el leaderboard exige la API key privada del participante, y la guía
es explícita: *"Las credenciales administrativas de Supabase y la API key de
submissions nunca deben llegar al navegador"*. Cualquiera que abra la página
podría leerla desde el código fuente. Se prefiere perder un panel antes que
exponer la credencial con la que se firman nuestras entregas.

## Seguridad

La página usa la **llave publicable** de Supabase, que es de solo lectura. No
es un descuido que esté en el código: es su propósito. Está verificado en la
práctica —lectura devuelve `200`, escritura devuelve `401`— gracias a las
políticas RLS descritas en [`docs/entity-relation.md`](../docs/entity-relation.md).

## Desplegado en

**https://pulso-transmi-one.vercel.app** — se actualiza solo en cada push a `main`.

## Cómo se desplegó

No tiene build ni dependencias: es un archivo estático.

1. Entrar a https://vercel.com y conectar la cuenta de GitHub.
2. **Add New → Project** e importar `kevin-nieto-callejas/pulso-transmi`.
3. En **Root Directory**, elegir `dashboard`.
4. Framework Preset: **Other**. Dejar los comandos de build vacíos.
5. Deploy.

Cada push a `main` vuelve a desplegar automáticamente.

## Probar en local

```bash
python -m http.server 8000 --directory dashboard
# abrir http://localhost:8000
```

## Estados vacíos

Antes de que empiece la competencia no hay ciclos evaluados, así que la
gráfica de accuracy y la de estaciones muestran un mensaje explicando qué
falta en vez de un gráfico roto. Eso es intencional: un panel que miente
cuando no tiene datos es peor que uno que dice que no los tiene.

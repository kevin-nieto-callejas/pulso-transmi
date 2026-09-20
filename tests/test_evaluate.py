import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate  # noqa: E402


def _historial(accuracies, station_id=None, minutos=60):
    """Historial de ciclos: station_id None = fila total del ciclo.

    Los ciclos se colocan hacia atras desde ahora, separados `minutos`.
    """
    ahora = pd.Timestamp.now(tz="UTC")
    n = len(accuracies)
    return pd.DataFrame([
        {"cycle_id": f"cyc_{i}", "station_id": station_id, "accuracy": a,
         "computed_at": (ahora - pd.Timedelta(minutes=minutos * (n - 1 - i))).isoformat()}
        for i, a in enumerate(accuracies)
    ])


def _dia_completo(accuracy_media, n=26):
    """Un dia de ciclos alrededor de una media, con la variacion por hora que
    se midio de verdad en el simulacro (dia ~85, madrugada ~72)."""
    import math
    return [accuracy_media + 7 * math.sin(2 * math.pi * i / 24) for i in range(n)]


def test_solo_se_evaluan_predicciones_con_realidad_conocida() -> None:
    """Un ciclo se evalua progresivamente: +15 min se resuelve antes que +60,
    asi que las predicciones sin observacion real todavia no cuentan."""
    predicciones = pd.DataFrame([
        {"id": 1, "cycle_id": "cyc_1", "station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00", "predicted_value": 100.0},
        {"id": 2, "cycle_id": "cyc_1", "station_id": "03000", "target_at": "2026-09-20T11:00:00+00:00", "predicted_value": 120.0},
    ])
    observaciones = pd.DataFrame([
        {"station_id": "03000", "observed_at": "2026-09-20T10:15:00+00:00", "demand": 90},
    ])

    resultado = evaluate.emparejar_con_la_realidad(predicciones, observaciones)

    assert len(resultado) == 1
    assert resultado.iloc[0]["id"] == 1
    assert resultado.iloc[0]["absolute_error"] == 10.0


def test_el_accuracy_del_ciclo_no_pondera_por_volumen() -> None:
    """La formula oficial promedia las 12 estaciones SIN ponderar, para que
    una estacion grande no tape a una pequena. Aqui una estacion de volumen
    alto acierta y una de volumen bajo falla feo: el total debe quedar a
    medio camino, no arrastrado por la grande."""
    evaluaciones = pd.DataFrame([
        {"cycle_id": "cyc_1", "station_id": "GRANDE", "predicted_value": 1000.0, "actual_value": 1000.0},
        {"cycle_id": "cyc_1", "station_id": "PEQUENA", "predicted_value": 10.0, "actual_value": 20.0},
    ])

    metricas = evaluate.metricas_por_ciclo(evaluaciones)
    total = metricas[metricas["station_id"].isna()].iloc[0]["accuracy"]
    grande = metricas[metricas["station_id"] == "GRANDE"].iloc[0]["accuracy"]
    pequena = metricas[metricas["station_id"] == "PEQUENA"].iloc[0]["accuracy"]

    assert grande == 100.0
    assert pequena == 50.0
    assert total == 75.0  # promedio simple; ponderado por volumen daria ~99


def _historial_normal(n=60, media=85.74, sigma=2.60, semilla=7):
    """Historial con la variabilidad REAL medida sobre 2.229 ciclos.

    Los ciclos NO son independientes: arrastran el estado del dia (una tarde
    dificil lo es entera). Generarlos independientes daba ventanas de 24 con
    desviacion 0.19, cuando la medida de verdad es 1.51 — un test con datos
    demasiado limpios habria dado por bueno un detector hipersensible.
    """
    import math
    import random
    rng = random.Random(semilla)
    valores, arrastre = [], 0.0
    for i in range(n):
        arrastre = 0.75 * arrastre + rng.gauss(0, sigma * 0.55)   # memoria entre ciclos
        ciclo_diario = 1.2 * math.sin(2 * math.pi * i / 24)        # franja horaria
        valores.append(media + arrastre + ciclo_diario)
    return _historial(valores)


def test_un_ciclo_malo_aislado_no_dispara_alarma() -> None:
    """Medido sobre 2.229 ciclos: uno suelto baja hasta 57.79 sin que nada
    falle. El 2.2% queda por debajo de 80 solo por azar."""
    hist = _historial_normal()
    hist.loc[hist.index[-1], "accuracy"] = 57.79

    assert evaluate.detectar_degradacion(hist) is None


def test_la_variacion_normal_no_dispara_alarma() -> None:
    """Un historial con el ruido tipico no debe generar ninguna senal: si lo
    hiciera, tendriamos alarmas constantes durante toda la competencia."""
    for semilla in range(10):
        hist = _historial_normal(semilla=semilla)
        assert evaluate.detectar_degradacion(hist) is None, f"falsa alarma con semilla {semilla}"


def test_caida_sostenida_si_dispara_alarma() -> None:
    """Una caida real y mantenida -no un ciclo malo- si debe detectarse."""
    hist = _historial_normal(n=80)
    # Las ultimas 24 horas caen 6 puntos: mas de 3 desviaciones de la ventana.
    hist.loc[hist.index[-24:], "accuracy"] = hist["accuracy"].iloc[-24:] - 6.0

    senal = evaluate.detectar_degradacion(hist)

    assert senal is not None
    assert senal.tipo == "performance"
    assert "desviaciones por debajo" in senal.descripcion


def test_no_se_juzga_sin_historial_de_referencia() -> None:
    """Drift es cambio respecto a como venia. Sin historial propio no hay
    contra que comparar, y opinar seria inventar."""
    assert evaluate.detectar_degradacion(_historial_normal(n=20)) is None


def test_data_drift_compara_cada_estacion_contra_si_misma() -> None:
    """Entre estaciones hay mas de 3x de diferencia en demanda media, asi que
    un umbral comun no significaria nada. La estacion tranquila que se
    dispara debe detectarse aunque su demanda siga siendo menor que la de una
    estacion grande que no cambio."""
    referencia = pd.DataFrame(
        [{"station_id": "TRANQUILA", "demand": d} for d in [10, 12, 11, 9, 10, 11]]
        + [{"station_id": "GRANDE", "demand": d} for d in [500, 520, 480, 510, 495, 505]]
    )
    reciente = pd.DataFrame([
        {"station_id": "TRANQUILA", "demand": 60},   # se disparo respecto a lo suyo
        {"station_id": "GRANDE", "demand": 505},     # normal
    ])

    senales = evaluate.detectar_cambio_en_datos(reciente, referencia)

    assert len(senales) == 1
    assert "TRANQUILA" in senales[0].descripcion
    assert senales[0].tipo == "data"


def test_se_detectan_ciclos_sin_entregar() -> None:
    ciclos = pd.DataFrame([{"cycle_id": "cyc_1"}, {"cycle_id": "cyc_2"}, {"cycle_id": "cyc_3"}])
    submissions = pd.DataFrame([{"cycle_id": "cyc_1"}])

    senal = evaluate.detectar_falla_operacional(ciclos, submissions)

    assert senal is not None
    assert senal.tipo == "operational"
    assert "2 ciclo(s)" in senal.descripcion


def test_la_falla_operacional_tiene_prioridad_sobre_reentrenar() -> None:
    """Un accuracy bajo porque no entregamos no se arregla reentrenando."""
    senales = [
        evaluate.Senal("performance", "cae el accuracy", 70.0, 84.0, ""),
        evaluate.Senal("operational", "ciclos sin entregar", 2.0, 0.0, ""),
    ]

    decision = evaluate.decidir(senales, observaciones_nuevas=99999, horas_desde_entrenamiento=48)

    assert decision.startswith("INVESTIGAR (operacion)")


def test_no_se_reentrena_sin_datos_nuevos_suficientes() -> None:
    """Reentrenar con menos de un dia de datos nuevos da practicamente el
    mismo modelo con otro nombre."""
    senales = [evaluate.Senal("performance", "cae el accuracy", 70.0, 84.0, "")]

    decision = evaluate.decidir(senales, observaciones_nuevas=200, horas_desde_entrenamiento=48)

    assert decision.startswith("INVESTIGAR")
    assert "200 observaciones nuevas" in decision


def test_se_reentrena_con_degradacion_datos_y_tiempo_suficientes() -> None:
    senales = [evaluate.Senal("performance", "cae el accuracy", 70.0, 84.0, "")]

    decision = evaluate.decidir(senales, observaciones_nuevas=5000, horas_desde_entrenamiento=48)

    assert decision.startswith("REENTRENAR")


def test_tablas_vacias_no_rompen_el_monitoreo() -> None:
    """Regresion real: PostgREST devuelve una tabla vacia SIN columnas, asi
    que acceder a historial['station_id'] reventaba. Antes de la competencia
    no hay nada evaluado todavia, y eso es lo normal, no un error."""
    vacio = pd.DataFrame()

    assert evaluate.detectar_degradacion(vacio, metrica_esperada=86.61) is None
    assert evaluate.accuracy_movil(vacio) is None
    assert evaluate.detectar_falla_operacional(vacio, vacio) is None
    assert evaluate.detectar_cambio_en_datos(vacio, vacio) == []
    assert evaluate.metricas_por_ciclo(vacio).empty


def test_sin_senales_se_mantiene() -> None:
    assert evaluate.decidir([], observaciones_nuevas=5000, horas_desde_entrenamiento=48).startswith("MANTENER")

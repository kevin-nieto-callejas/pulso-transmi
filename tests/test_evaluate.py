import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate  # noqa: E402


def _historial(accuracies, station_id=None):
    """Historial de ciclos: station_id None = fila total del ciclo."""
    base = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=len(accuracies))
    return pd.DataFrame([
        {"cycle_id": f"cyc_{i}", "station_id": station_id,
         "accuracy": a, "computed_at": (base + pd.Timedelta(hours=i)).isoformat()}
        for i, a in enumerate(accuracies)
    ])


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


def test_un_solo_ciclo_malo_no_dispara_alarma() -> None:
    """Una hora punta atipica, un partido o un aguacero producen un ciclo
    malo. Reentrenar por eso seria reaccionar al ruido."""
    historial = _historial([86.5, 86.2, 70.0])  # solo el ultimo esta mal

    assert evaluate.detectar_degradacion(historial, metrica_esperada=86.61) is None


def test_degradacion_sostenida_si_dispara_alarma() -> None:
    historial = _historial([86.5, 80.0, 79.5, 78.0])  # los 3 ultimos, caidos

    senal = evaluate.detectar_degradacion(historial, metrica_esperada=86.61)

    assert senal is not None
    assert senal.tipo == "performance"
    assert "3 ciclos seguidos" in senal.descripcion


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

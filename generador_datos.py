# -*- coding: utf-8 -*-
"""
generador_datos.py
-------------------
Generador de datos sintéticos para el Sistema Predictivo de Rendimiento
Académico enfocado en estudiantes universitarios que trabajan.

Diseño de las correlaciones (para que la predicción sea "hiperrealista"
y lógicamente consistente):

  - A mayor horas_trabajo_semanal y menor horas_estudio_semanal, MENOR
    examen_final y nota_final (mayor probabilidad de Riesgo/Deserción).
  - A mayor entendimiento_curso y asistencia, MAYOR nota_final (esto
    compensa parcialmente el efecto negativo de trabajar, evitando que
    el sistema "condene" automáticamente a quien trabaja mucho pero es
    eficiente - ver módulo de Ética en app.py).
  - nivel_estres sube con horas_trabajo y baja con horas_sueño.
  - carga_familiar (personas que dependen económicamente del alumno)
    incrementa el estrés y reduce las horas de estudio disponibles.
  - T1 y T2 están correlacionadas entre sí (consistencia del alumno en
    el ciclo) y con el desempeño futuro en el examen final.

Variables NUEVAS agregadas más allá de lo mínimo pedido, pensadas para
el perfil "estudiante que trabaja":
  - nivel_estres (1-10)
  - horas_sueno (horas de sueño diarias)
  - carga_familiar (num. de personas que dependen económicamente de él/ella)
  - tiempo_transporte (horas diarias en traslado casa-trabajo-universidad)
  - tiene_beca (0/1)
  - modalidad_trabajo (Presencial / Remoto / Híbrido) -> afecta tiempo_transporte
  - num_cursos_matriculados
  - salud_mental_autopercibida (1-10, autoevaluación, distinta de estrés)
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generar_dataset(n_muestras: int = 6000) -> pd.DataFrame:
    """Genera un DataFrame de n_muestras estudiantes con variables
    correlacionadas de forma realista."""

    # ------------------------------------------------------------------
    # 1. Variables demográficas / contextuales base
    # ------------------------------------------------------------------
    edad = rng.integers(18, 33, size=n_muestras)
    ciclo = rng.integers(1, 11, size=n_muestras)  # ciclo académico 1-10

    # 20% de la muestra son estudiantes que NO trabajan (grupo de contraste
    # para que los modelos aprendan el efecto real de trabajar, no solo
    # memoricen "todos trabajan").
    trabaja = rng.choice([0, 1], size=n_muestras, p=[0.20, 0.80])
    horas_trabajo_semanal = np.where(
        trabaja == 1,
        _clip(rng.normal(30, 8, n_muestras), 8, 48),
        0,
    ).round(1)

    modalidad_trabajo = rng.choice(
        ["Presencial", "Remoto", "Hibrido"], size=n_muestras, p=[0.55, 0.25, 0.20]
    )
    modalidad_trabajo = np.where(trabaja == 0, "No aplica", modalidad_trabajo)

    # tiempo de transporte: mayor si es presencial/hibrido y depende de horas trabajo
    base_transporte = np.select(
        [modalidad_trabajo == "Presencial", modalidad_trabajo == "Hibrido"],
        [rng.normal(1.6, 0.5, n_muestras), rng.normal(0.9, 0.4, n_muestras)],
        default=rng.normal(0.3, 0.2, n_muestras),
    )
    tiempo_transporte = _clip(base_transporte, 0, 4).round(2)

    carga_familiar = rng.choice([0, 1, 2, 3, 4], size=n_muestras,
                                 p=[0.45, 0.25, 0.15, 0.10, 0.05])

    tiene_beca = rng.choice([0, 1], size=n_muestras, p=[0.75, 0.25])
    num_cursos_matriculados = rng.integers(3, 8, size=n_muestras)

    # ------------------------------------------------------------------
    # 2. Hábitos de estudio y bienestar (correlacionados con trabajo)
    # ------------------------------------------------------------------
    # Horas de estudio: penalizadas por horas de trabajo y carga familiar,
    # pero con variabilidad individual (algunos gestionan mejor su tiempo).
    disciplina_individual = rng.normal(0, 1, n_muestras)  # rasgo latente
    horas_estudio_semanal = _clip(
        22
        - 0.28 * horas_trabajo_semanal
        - 1.1 * carga_familiar
        - 1.4 * tiempo_transporte
        + 3.0 * disciplina_individual
        + rng.normal(0, 2.5, n_muestras),
        0, 40,
    ).round(1)

    # Horas de sueño: penalizadas por trabajo + estudio combinados
    horas_sueno = _clip(
        7.5
        - 0.03 * horas_trabajo_semanal
        - 0.02 * horas_estudio_semanal
        - 0.15 * carga_familiar
        + rng.normal(0, 0.8, n_muestras),
        3.5, 9.5,
    ).round(1)

    # Nivel de estrés: sube con trabajo, carga familiar y transporte;
    # baja con sueño y disciplina.
    nivel_estres = _clip(
        4.0
        + 0.09 * horas_trabajo_semanal
        + 0.7 * carga_familiar
        + 0.6 * tiempo_transporte
        - 0.35 * horas_sueno
        - 0.5 * disciplina_individual
        + rng.normal(0, 1.2, n_muestras),
        1, 10,
    ).round(1)

    # Salud mental autopercibida: inversamente relacionada al estrés,
    # pero no es un espejo exacto (aporta variables no linealmente
    # redundantes al modelo).
    salud_mental_autopercibida = _clip(
        9.0 - 0.55 * nivel_estres + rng.normal(0, 1.0, n_muestras), 1, 10
    ).round(1)

    # Asistencia %: penalizada por trabajo y transporte, mejorada por beca
    # (compromiso) y disciplina.
    asistencia_pct = _clip(
        92
        - 0.35 * horas_trabajo_semanal
        - 3.0 * tiempo_transporte
        + 4.0 * tiene_beca
        + 2.5 * disciplina_individual
        + rng.normal(0, 6, n_muestras),
        30, 100,
    ).round(1)

    # Entendimiento del curso (1-10, autoevaluado): depende de horas de
    # estudio y asistencia, con algo de "talento" individual aleatorio.
    talento = rng.normal(0, 1, n_muestras)
    entendimiento_curso = _clip(
        4.5
        + 0.09 * horas_estudio_semanal
        + 0.02 * asistencia_pct
        + 1.1 * talento
        - 0.15 * nivel_estres
        + rng.normal(0, 1.0, n_muestras),
        1, 10,
    ).round(1)

    # ------------------------------------------------------------------
    # 3. Notas T1 y T2 (0-20, escala peruana)
    # ------------------------------------------------------------------
    T1 = _clip(
        6.0
        + 0.25 * entendimiento_curso
        + 0.05 * asistencia_pct
        + 0.55 * talento
        - 0.10 * nivel_estres
        + rng.normal(0, 1.6, n_muestras),
        0, 20,
    ).round(1)

    # T2 correlacionado con T1 (consistencia) + deriva propia
    T2 = _clip(
        0.75 * T1
        + 0.20 * entendimiento_curso
        + 0.03 * asistencia_pct
        - 0.08 * nivel_estres
        + rng.normal(0, 1.8, n_muestras),
        0, 20,
    ).round(1)

    # ------------------------------------------------------------------
    # 4. Examen final (lo que el modelo de Regresión Lineal predecirá
    #    cuando el usuario NO conozca aún su nota final)
    # ------------------------------------------------------------------
    examen_final = _clip(
        0.35 * T2
        + 0.20 * T1
        + 0.22 * entendimiento_curso
        + 0.05 * asistencia_pct
        - 0.11 * horas_trabajo_semanal * (1 - 0.35 * (horas_estudio_semanal > 15))
        + 0.18 * horas_estudio_semanal
        - 0.10 * nivel_estres
        + 0.35 * horas_sueno
        - 0.6 * carga_familiar
        + 0.9 * talento
        + rng.normal(0, 1.7, n_muestras),
        0, 20,
    ).round(1)

    # Nota final ponderada (criterio típico universitario): T1 20%, T2 30%,
    # Examen Final 50%.
    nota_final = _clip(
        0.20 * T1 + 0.30 * T2 + 0.50 * examen_final, 0, 20
    ).round(2)

    # ------------------------------------------------------------------
    # 5. Situación académica (variable objetivo de clasificación)
    #    Se construye con reglas que evitan sesgar automáticamente a
    #    quienes trabajan muchas horas: lo que pesa es el DESEMPEÑO
    #    (nota, asistencia) y no la condición laboral en sí misma.
    # ------------------------------------------------------------------
    situacion = np.empty(n_muestras, dtype=object)
    riesgo_estructural = (
        (nota_final < 11).astype(int)
        + (asistencia_pct < 60).astype(int)
        + (nivel_estres >= 8).astype(int)
        + (carga_familiar >= 3).astype(int)
    )

    for i in range(n_muestras):
        if nota_final[i] >= 11 and asistencia_pct[i] >= 55:
            situacion[i] = "Aprobado"
        elif nota_final[i] < 8 and riesgo_estructural[i] >= 3:
            situacion[i] = "Desercion"
        else:
            situacion[i] = "Riesgo"

    df = pd.DataFrame({
        "edad": edad,
        "ciclo": ciclo,
        "trabaja": trabaja,
        "horas_trabajo_semanal": horas_trabajo_semanal,
        "modalidad_trabajo": modalidad_trabajo,
        "tiempo_transporte": tiempo_transporte,
        "carga_familiar": carga_familiar,
        "tiene_beca": tiene_beca,
        "num_cursos_matriculados": num_cursos_matriculados,
        "horas_estudio_semanal": horas_estudio_semanal,
        "horas_sueno": horas_sueno,
        "nivel_estres": nivel_estres,
        "salud_mental_autopercibida": salud_mental_autopercibida,
        "asistencia_pct": asistencia_pct,
        "entendimiento_curso": entendimiento_curso,
        "T1": T1,
        "T2": T2,
        "examen_final": examen_final,
        "nota_final": nota_final,
        "situacion_academica": situacion,
    })

    # ------------------------------------------------------------------
    # 6. Inyección controlada de outliers (para que el pipeline de
    #    preprocesamiento tenga algo real que limpiar)
    # ------------------------------------------------------------------
    n_outliers = int(n_muestras * 0.015)
    idx_outliers = rng.choice(n_muestras, n_outliers, replace=False)
    df.loc[idx_outliers, "horas_trabajo_semanal"] = rng.choice(
        [0, 90, 95, 100], size=n_outliers
    )
    idx_outliers2 = rng.choice(n_muestras, n_outliers, replace=False)
    df.loc[idx_outliers2, "horas_estudio_semanal"] = rng.choice(
        [0, 80, 85], size=n_outliers
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generar_dataset(6000)
    ruta = os.path.join("data", "dataset_estudiantes.csv")
    df.to_csv(ruta, index=False)

    print(f"Dataset generado con {len(df)} registros -> {ruta}")
    print("\nDistribución de 'situacion_academica':")
    print(df["situacion_academica"].value_counts())
    print("\nEstadísticas descriptivas:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    print("\nCorrelación clave (validación de lógica del negocio):")
    print(df[["horas_trabajo_semanal", "horas_estudio_semanal",
              "nivel_estres", "nota_final"]].corr()["nota_final"])

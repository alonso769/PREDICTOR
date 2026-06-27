"""
Generador de dataset sintético - Estudiantes UPN Modalidad para Gente que Trabaja
Incluye features de notas T1/T2 para predicción integrada.
"""

import pandas as pd
import numpy as np
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generar_estudiante():
    horas_trabajo       = np.random.choice([0, 4, 8, 9, 10, 12], p=[0.05, 0.10, 0.35, 0.20, 0.20, 0.10])
    cursos_matriculados = np.random.randint(2, 6)
    semestre            = np.random.randint(1, 11)

    if horas_trabajo >= 10:
        asistencia = np.random.normal(60, 15)
    elif horas_trabajo >= 8:
        asistencia = np.random.normal(72, 12)
    else:
        asistencia = np.random.normal(82, 10)
    asistencia = np.clip(asistencia, 20, 100)

    horas_estudio = max(1, np.random.normal(14 - horas_trabajo * 0.6, 3))
    horas_estudio = round(np.clip(horas_estudio, 1, 30), 1)

    base_promedio = (
        asistencia * 0.10
        + horas_estudio * 0.25
        - horas_trabajo * 0.20
        + (semestre * 0.3)
    )
    promedio = np.random.normal(base_promedio * 0.18, 1.5)
    promedio = round(np.clip(promedio, 5.0, 20.0), 2)

    p_jale        = max(0, (0.5 - promedio / 40))
    cursos_jalados = np.random.binomial(semestre, p_jale)
    cursos_jalados = min(cursos_jalados, semestre * cursos_matriculados)

    deuda            = np.random.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.15, 0.10])
    internet_estable = np.random.choice([0, 1], p=[0.20, 0.80])
    participacion    = max(1, min(5, int(np.random.normal(3 + (asistencia - 60) / 30, 0.8))))
    entrega_trabajos = round(np.clip(np.random.normal(asistencia + 5, 8), 20, 100), 1)
    distancia_campus = min(round(np.random.exponential(15), 1), 80)
    apoyo_familiar   = np.random.randint(1, 6)
    uso_tutoria      = np.random.choice([0, 1], p=[0.60, 0.40])
    modalidad_trabajo= np.random.choice(['presencial','remoto','mixto','sin_trabajo'], p=[0.40,0.25,0.25,0.10])
    turno            = np.random.choice(['noche','sabado_domingo'], p=[0.65, 0.35])

    # ── NUEVAS FEATURES: Notas T1/T2 sintéticas ──────────────────────
    # Se generan coherentes con el promedio histórico del estudiante
    ruido_t1 = np.random.normal(0, 2.0)
    ruido_t2 = np.random.normal(0, 2.0)
    promedio_t1 = round(np.clip(promedio + ruido_t1, 0, 20), 2)
    promedio_t2 = round(np.clip(promedio + ruido_t2, 0, 20), 2)

    # Cursos que probablemente jalará este ciclo (nota T1 < 11)
    cursos_en_riesgo_notas = int(round(
        cursos_matriculados * max(0, (11 - promedio_t1) / 11)
    ))
    cursos_en_riesgo_notas = np.clip(cursos_en_riesgo_notas, 0, cursos_matriculados)

    # Porcentaje de créditos en riesgo (aprox. 3 créditos por curso promedio)
    pct_creditos_riesgo = round(cursos_en_riesgo_notas / max(cursos_matriculados, 1), 2)

    # ─── ETIQUETA ────────────────────────────────────────────────────
    score = (
        promedio * 3.0
        + (asistencia / 100) * 25
        + (entrega_trabajos / 100) * 20
        + horas_estudio * 0.8
        - horas_trabajo * 0.5
        - cursos_jalados * 2.0
        + apoyo_familiar * 1.5
        + participacion * 1.5
        + uso_tutoria * 3
        - deuda * 2
        + promedio_t1 * 1.5          # T1 influye en la situación
        + promedio_t2 * 1.0          # T2 también
        - cursos_en_riesgo_notas * 3 # Más cursos en riesgo → peor situación
        - pct_creditos_riesgo * 10
    )

    if score >= 85:
        situacion = 'APROBADO_REGULAR'
    elif score >= 68:
        situacion = 'EN_RIESGO'
    elif score >= 50:
        situacion = 'RIESGO_ALTO'
    else:
        situacion = 'DESERCION_PROBABLE'

    return {
        'semestre':                  semestre,
        'cursos_matriculados':       cursos_matriculados,
        'horas_trabajo_semanal':     horas_trabajo,
        'modalidad_trabajo':         modalidad_trabajo,
        'turno_clases':              turno,
        'asistencia_pct':            round(asistencia, 1),
        'horas_estudio_semanal':     horas_estudio,
        'promedio_ponderado':        promedio,
        'cursos_jalados_acumulados': cursos_jalados,
        'entrega_trabajos_pct':      entrega_trabajos,
        'participacion_clase':       participacion,
        'distancia_campus_km':       distancia_campus,
        'internet_estable':          internet_estable,
        'apoyo_familiar':            apoyo_familiar,
        'uso_tutoria':               uso_tutoria,
        'deuda_pension':             deuda,
        # ── Nuevas features de notas ──
        'promedio_t1':               promedio_t1,
        'promedio_t2':               promedio_t2,
        'cursos_en_riesgo_notas':    int(cursos_en_riesgo_notas),
        'pct_creditos_riesgo':       pct_creditos_riesgo,
        'situacion_academica':       situacion,
    }


def generar_dataset(n=1200):
    datos = [generar_estudiante() for _ in range(n)]
    return pd.DataFrame(datos)


if __name__ == '__main__':
    ruta = os.path.join(os.path.dirname(__file__), 'data', 'estudiantes_upn.csv')
    if os.path.exists(ruta):
        df_existente = pd.read_csv(ruta)
        nuevos = int(input(f"Dataset actual: {len(df_existente)} registros. ¿Cuántos casos nuevos agregar? "))
        df_nuevo = generar_dataset(nuevos)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
        print(f"→ Dataset actualizado: {len(df_final)} registros")
    else:
        df_final = generar_dataset(1200)
        print(f"→ Dataset creado: {len(df_final)} registros")
    df_final.to_csv(ruta, index=False)
    print(f"✅ Guardado en: {ruta}")
    print(df_final['situacion_academica'].value_counts())

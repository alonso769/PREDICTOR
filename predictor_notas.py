"""
predictor_notas.py
Módulo de predicción de aprobación por evaluaciones (T1, T2, Final)
por curso individual y por semestre completo.
Universidad Privada del Norte - Ingeniería de Sistemas Computacionales (WA 2023)
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CATÁLOGO DE CURSOS - Ingeniería de Sistemas UPN WA (Periodo 223000)
# Extraído de la malla oficial Adulto Trabajador
# ─────────────────────────────────────────────
CURSOS_UPN = {
    # ── CICLO 1 ───────────────────────────────
    'Complemento Matemático para Ingenieros':       {'codigo': 'MATH1002',  'tipo': 'matematica', 'ciclo': 1, 'creditos': 4},
    'Introducción a la Ing. de Sistemas':           {'codigo': 'SOIT1101A', 'tipo': 'carrera',    'ciclo': 1, 'creditos': 2},
    'Comunicación 1':                               {'codigo': 'LENG1001',  'tipo': 'letras',     'ciclo': 1, 'creditos': 4},
    'Desarrollo del Talento':                       {'codigo': 'RRHH1101',  'tipo': 'letras',     'ciclo': 1, 'creditos': 2},
    'Ciudadanía Global':                            {'codigo': 'HUMA1111',  'tipo': 'letras',     'ciclo': 1, 'creditos': 2},
    
    # ── CICLO 2 ───────────────────────────────
    'Matemática Básica para Ingeniería':            {'codigo': 'MATH1005',  'tipo': 'matematica', 'ciclo': 2, 'creditos': 3},
    'Comunicación 2':                               {'codigo': 'LENG1002',  'tipo': 'letras',     'ciclo': 2, 'creditos': 4},
    'Fundamentos de Algoritmos':                    {'codigo': 'COMP131A',  'tipo': 'carrera',    'ciclo': 2, 'creditos': 2},
    'Mecánica Oscilación y Ondas':                  {'codigo': 'FISI201A',  'tipo': 'matematica', 'ciclo': 2, 'creditos': 2},
    'Metodología Universitaria':                    {'codigo': 'INVE1101',  'tipo': 'letras',     'ciclo': 2, 'creditos': 2},

    # ── CICLO 3 ───────────────────────────────
    'Herramientas Informáticas':                    {'codigo': 'INFO1120A', 'tipo': 'carrera',    'ciclo': 3, 'creditos': 2},
    'Cálculo 1':                                    {'codigo': 'MATH1101A', 'tipo': 'matematica', 'ciclo': 3, 'creditos': 4},
    'Fundamentos de Programación':                  {'codigo': 'SOFT1101A', 'tipo': 'carrera',    'ciclo': 3, 'creditos': 4},
    'Matemática Discreta':                          {'codigo': 'MATH1103',  'tipo': 'matematica', 'ciclo': 3, 'creditos': 4},
    'Electricidad Magnetismo y Óptica':             {'codigo': 'FISI202A',  'tipo': 'matematica', 'ciclo': 3, 'creditos': 2},

    # ── CICLO 4 ───────────────────────────────
    'Probabilidad y Estadística':                   {'codigo': 'STAT1203A', 'tipo': 'matematica', 'ciclo': 4, 'creditos': 2},
    'Estructura de Datos':                          {'codigo': 'SOFT1201A', 'tipo': 'carrera',    'ciclo': 4, 'creditos': 4},
    'Optimización y Simulación':                    {'codigo': 'OPER1305',  'tipo': 'matematica', 'ciclo': 4, 'creditos': 4},
    'Cálculo 2':                                    {'codigo': 'MATH1201A', 'tipo': 'matematica', 'ciclo': 4, 'creditos': 4},
    'Electrónica Digital':                          {'codigo': 'FISI203A',  'tipo': 'carrera',    'ciclo': 4, 'creditos': 2},

    # ── CICLO 5 ───────────────────────────────
    'Análisis de Algoritmos y Estrategias':         {'codigo': 'COMP1301A', 'tipo': 'carrera',    'ciclo': 5, 'creditos': 2},
    'Comunicación 3':                               {'codigo': 'LENG1003',  'tipo': 'letras',     'ciclo': 5, 'creditos': 4},
    'Base de Datos':                                {'codigo': 'ISOF1201',  'tipo': 'carrera',    'ciclo': 5, 'creditos': 4},
    'Responsabilidad Social':                       {'codigo': 'HUMA1111',  'tipo': 'letras',     'ciclo': 5, 'creditos': 2},
    'Técnicas de Programación Orientada a Objetos': {'codigo': 'SOFT1202A', 'tipo': 'carrera',    'ciclo': 5, 'creditos': 4},

    # ── CICLO 6 ───────────────────────────────
    'Base de Datos Avanzadas y Big Data':           {'codigo': 'ISOFT1304', 'tipo': 'carrera',    'ciclo': 6, 'creditos': 4},
    'Metodología de la Investigación':              {'codigo': 'INVE1301',  'tipo': 'letras',     'ciclo': 6, 'creditos': 3},
    'Computación Gráfica y Visual':                 {'codigo': 'COMP1401A', 'tipo': 'carrera',    'ciclo': 6, 'creditos': 3},
    'Modelamiento y Análisis de Software':          {'codigo': 'SOFT1301',  'tipo': 'carrera',    'ciclo': 6, 'creditos': 2},
    'Proyecto Social':                              {'codigo': 'HUMA1408',  'tipo': 'letras',     'ciclo': 6, 'creditos': 2},
    'Arquitectura del Computador':                  {'codigo': 'ITEC1201A', 'tipo': 'carrera',    'ciclo': 6, 'creditos': 3},

    # ── CICLO 7 ───────────────────────────────
    'Sistemas Operativos':                          {'codigo': 'ITEC1301A', 'tipo': 'carrera',    'ciclo': 7, 'creditos': 4},
    'Interacción Humano Computador':                {'codigo': 'COMP1302',  'tipo': 'carrera',    'ciclo': 7, 'creditos': 4},
    'Empleabilidad':                                {'codigo': 'RRHH1303',  'tipo': 'letras',     'ciclo': 7, 'creditos': 2},
    'Diseño y Arquitectura de Software':            {'codigo': 'SOFT1401A', 'tipo': 'carrera',    'ciclo': 7, 'creditos': 4},
    'Redes 1':                                      {'codigo': 'ITEC1301',  'tipo': 'carrera',    'ciclo': 7, 'creditos': 3},

    # ── CICLO 8 ───────────────────────────────
    'Soluciones Web y Aplicaciones Distribuidas':   {'codigo': 'SOFT1402A', 'tipo': 'carrera',    'ciclo': 8, 'creditos': 4},
    'Redes 2':                                      {'codigo': 'ITEC1402A', 'tipo': 'carrera',    'ciclo': 8, 'creditos': 4},
    'Prácticas Preprofesionales':                   {'codigo': 'INVE1416A', 'tipo': 'letras',     'ciclo': 8, 'creditos': 2},
    'Calidad y Pruebas de Software':                {'codigo': 'ISOF1401A', 'tipo': 'carrera',    'ciclo': 8, 'creditos': 4},
    'Taller de Robótica':                           {'codigo': 'ROBS1405',  'tipo': 'carrera',    'ciclo': 8, 'creditos': 4},

    # ── CICLO 9 ───────────────────────────────
    'Sistemas Inteligentes y Machine Learning':     {'codigo': 'SIST1504',  'tipo': 'carrera',    'ciclo': 9, 'creditos': 4},
    'Desarrollo de Aplicaciones Móviles':           {'codigo': 'SIST1402',  'tipo': 'carrera',    'ciclo': 9, 'creditos': 4},
    'Administración de Proyectos de Software':      {'codigo': 'ISOF1402A', 'tipo': 'carrera',    'ciclo': 9, 'creditos': 4},
    'Capstone Project Sistemas':                    {'codigo': 'INVE1435',  'tipo': 'carrera',    'ciclo': 9, 'creditos': 4},
    'Seguridad Informática':                        {'codigo': 'INFO1523',  'tipo': 'carrera',    'ciclo': 9, 'creditos': 4},

    # ── CICLO 10 ──────────────────────────────
    'Trabajo de Investigación':                     {'codigo': 'TINV1501',  'tipo': 'letras',     'ciclo': 10, 'creditos': 4},
    
    # ── ELECTIVOS (Asignados a ciclo 8, 9 o 10 genéricamente) ──
    'Electivo: Desarrollo de Videojuegos':          {'codigo': 'ELEC1', 'tipo': 'carrera', 'ciclo': 8, 'creditos': 2},
    'Electivo: Redes Inalámbricas y Telecom.':      {'codigo': 'ITEC1520B', 'tipo': 'carrera', 'ciclo': 9, 'creditos': 2},
    'Electivo: Cloud Computing y Continuidad':      {'codigo': 'COMP1502',  'tipo': 'carrera', 'ciclo': 9, 'creditos': 2},
    'Electivo: Evolución y Config. de Software':    {'codigo': 'SOFT1521A', 'tipo': 'carrera', 'ciclo': 10,'creditos': 2},
    'Electivo: E-Business y Analítica Web':         {'codigo': 'INFO1502A', 'tipo': 'carrera', 'ciclo': 10,'creditos': 2},
    'Electivo: Gestión Procesos BPM y Estrategia':  {'codigo': 'MAGM1448',  'tipo': 'carrera', 'ciclo': 10,'creditos': 2},
}

# Pesos de evaluaciones UPN MGT (sobre 20)
PESO_T1     = 0.25
PESO_T2     = 0.35
PESO_FINAL  = 0.40
NOTA_MINIMA = 11.0   

DIFICULTAD = {
    'matematica': 1.20,   
    'carrera':    1.05,
    'letras':     0.90,   
}

def nota_maxima_posible(t1: float, t2: float | None) -> float:
    if t2 is None:
        return t1 * PESO_T1 + 20 * PESO_T2 + 20 * PESO_FINAL
    else:
        return t1 * PESO_T1 + t2 * PESO_T2 + 20 * PESO_FINAL

def nota_minima_final_requerida(t1: float, t2: float) -> float:
    acumulado = t1 * PESO_T1 + t2 * PESO_T2
    requerido = (NOTA_MINIMA - acumulado) / PESO_FINAL
    return round(requerido, 2)

def predecir_curso(
    nombre_curso: str,
    t1: float,
    t2: float | None,
    horas_estudio: float,
    asistencia: float,
    entiende_curso: int,       
    horas_trabajo: float,
    semestre_actual: int,
) -> dict:
    
    # Se ajusta el valor por defecto para que coincida con la nueva estructura
    info = CURSOS_UPN.get(nombre_curso, {'codigo': 'GEN000', 'tipo': 'carrera', 'ciclo': semestre_actual, 'creditos': 3})
    tipo = info['tipo']
    dif  = DIFICULTAD[tipo]

    if t2 is None:
        etapa = 'T1'
        nota_max = nota_maxima_posible(t1, None)
        puede_aprobar = nota_max >= NOTA_MINIMA
    else:
        etapa = 'T2'
        nota_max = nota_maxima_posible(t1, t2)
        puede_aprobar = nota_max >= NOTA_MINIMA
        min_final = nota_minima_final_requerida(t1, t2)

    base = (
        (t1 / 20) * 30
        + ((t2 / 20) * 25 if t2 is not None else 0)
        + (asistencia / 100) * 20
        + (horas_estudio / 30) * 15
        + (entiende_curso / 5) * 20
        - (horas_trabajo / 60) * 10
        - (info['ciclo'] <= 4 and tipo == 'matematica') * 5 # Actualizado a ciclo numérico
    ) / dif

    prob_aprobacion = round(min(max(base / 100, 0.0), 1.0), 3)

    factor_esfuerzo = (
        (horas_estudio / 20) * 0.3
        + (asistencia / 100) * 0.3
        + (entiende_curso / 5) * 0.4
    )
    if t2 is not None:
        acumulado = t1 * PESO_T1 + t2 * PESO_T2
        proyeccion_final_nota = min(
            20,
            acumulado + (factor_esfuerzo * 20 - 2 * dif) * PESO_FINAL
        )
    else:
        proyeccion_final_nota = min(20, t1 * 0.5 + factor_esfuerzo * 12)

    proyeccion_final_nota = round(max(0, proyeccion_final_nota), 2)

    if not puede_aprobar:
        situacion_curso = 'MATEMATICAMENTE_IMPOSIBLE'
    elif prob_aprobacion >= 0.75:
        situacion_curso = 'PROBABLE_APROBACION'
    elif prob_aprobacion >= 0.50:
        situacion_curso = 'EN_JUEGO'
    elif prob_aprobacion >= 0.30:
        situacion_curso = 'RIESGO_JALE'
    else:
        situacion_curso = 'PROBABLE_JALE'

    resultado = {
        'curso': nombre_curso,
        'codigo_banner': info['codigo'],
        'tipo': tipo,
        'ciclo': info['ciclo'],
        'creditos': info['creditos'],
        'etapa': etapa,
        'nota_t1': t1,
        'nota_t2': t2,
        'nota_maxima_posible': round(nota_max, 2),
        'proyeccion_nota_final': proyeccion_final_nota,
        'prob_aprobacion': prob_aprobacion,
        'situacion_curso': situacion_curso,
        'puede_aprobar': puede_aprobar,
    }
    if etapa == 'T2':
        resultado['nota_minima_final_requerida'] = min_final

    return resultado

def predecir_semestre(lista_cursos: list[dict]) -> dict:
    total = len(lista_cursos)
    if total == 0:
        return {}

    probables_jale = [c for c in lista_cursos if c['situacion_curso'] in ('PROBABLE_JALE', 'MATEMATICAMENTE_IMPOSIBLE')]
    en_juego       = [c for c in lista_cursos if c['situacion_curso'] == 'EN_JUEGO']
    probables_apr  = [c for c in lista_cursos if c['situacion_curso'] == 'PROBABLE_APROBACION']

    creditos_en_riesgo = sum(c['creditos'] for c in probables_jale)
    creditos_totales   = sum(c['creditos'] for c in lista_cursos)

    pct_riesgo = creditos_en_riesgo / creditos_totales if creditos_totales > 0 else 0

    if pct_riesgo == 0 and len(en_juego) == 0:
        situacion_semestre = 'SEMESTRE_CONTROLADO'
    elif pct_riesgo <= 0.25:
        situacion_semestre = 'SEMESTRE_MANEJABLE'
    elif pct_riesgo <= 0.50:
        situacion_semestre = 'SEMESTRE_EN_RIESGO'
    else:
        situacion_semestre = 'SEMESTRE_CRITICO'

    return {
        'total_cursos': total,
        'probables_aprobacion': len(probables_apr),
        'en_juego': len(en_juego),
        'probables_jale': len(probables_jale),
        'creditos_en_riesgo': creditos_en_riesgo,
        'creditos_totales': creditos_totales,
        'situacion_semestre': situacion_semestre,
        'detalle': lista_cursos,
    }

SITUACION_CURSO_ES = {
    'PROBABLE_APROBACION':      '✅ Probable aprobación',
    'EN_JUEGO':                 '⚠️ Puede salvarse',
    'RIESGO_JALE':              '🔴 Riesgo de jale',
    'PROBABLE_JALE':            '🚨 Probable jale',
    'MATEMATICAMENTE_IMPOSIBLE':'❌ Imposible aprobar',
}
SITUACION_CURSO_COLOR = {
    'PROBABLE_APROBACION':       '#2e7d32',
    'EN_JUEGO':                  '#f57f17',
    'RIESGO_JALE':               '#e65100',
    'PROBABLE_JALE':             '#b71c1c',
    'MATEMATICAMENTE_IMPOSIBLE': '#4a0000',
}
SITUACION_SEMESTRE_ES = {
    'SEMESTRE_CONTROLADO':  '✅ Semestre controlado',
    'SEMESTRE_MANEJABLE':   '⚠️ Semestre manejable',
    'SEMESTRE_EN_RIESGO':   '🔴 Semestre en riesgo',
    'SEMESTRE_CRITICO':     '🚨 Semestre crítico',
}
SITUACION_SEMESTRE_COLOR = {
    'SEMESTRE_CONTROLADO':  '#2e7d32',
    'SEMESTRE_MANEJABLE':   '#f57f17',
    'SEMESTRE_EN_RIESGO':   '#e65100',
    'SEMESTRE_CRITICO':     '#b71c1c',
}
RECOMENDACIONES_CURSO = {
    'PROBABLE_APROBACION': [
        'Mantén el ritmo de estudio actual.',
        'No bajes la guardia en el examen final.',
    ],
    'EN_JUEGO': [
        'Necesitas una buena nota en el final para asegurar.',
        'Prioriza este curso en tus horas de estudio esta semana.',
        'Asiste a las asesorías del docente antes del final.',
    ],
    'RIESGO_JALE': [
        'Busca tutoría de urgencia para este curso.',
        'Arma un grupo de estudio con compañeros que van bien.',
        'Habla con el docente para saber en qué reforzar.',
    ],
    'PROBABLE_JALE': [
        'Evalúa si conviene retirarte del curso antes de que cierre el plazo.',
        'Si no te retiras, enfoca TODO tu tiempo libre en este curso.',
        'Solicita apoyo al área de tutoría UPN esta semana.',
    ],
    'MATEMATICAMENTE_IMPOSIBLE': [
        'Matemáticamente ya no es posible aprobar aunque saques 20 en el final.',
        'Retírate formalmente del curso antes de la fecha límite para no afectar tu promedio.',
        'Planifica llevarlo el siguiente ciclo con más tiempo disponible.',
    ],
}
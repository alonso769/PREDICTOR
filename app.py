"""
Sistema Predictivo de Situación Académica
Universidad Privada del Norte - Modalidad para Gente que Trabaja
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_upn.pkl')
META_PATH  = os.path.join(BASE_DIR, 'models', 'modelo_meta.json')
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'estudiantes_upn.csv')

LABEL_ES = {
    'APROBADO_REGULAR':   '✅ Aprobado Regular',
    'EN_RIESGO':          '⚠️ En Riesgo',
    'RIESGO_ALTO':        '🔴 Riesgo Alto',
    'DESERCION_PROBABLE': '🚨 Deserción Probable',
}
LABEL_COLOR = {
    'APROBADO_REGULAR':   '#2e7d32',
    'EN_RIESGO':          '#f57f17',
    'RIESGO_ALTO':        '#e65100',
    'DESERCION_PROBABLE': '#b71c1c',
}
RECOMENDACIONES = {
    'APROBADO_REGULAR': [
        'Mantén tu ritmo de asistencia y estudio.',
        'Considera tomar talleres de certificación para potenciar tu perfil.',
        'Apoya a compañeros en riesgo a través de tutoría entre pares.',
    ],
    'EN_RIESGO': [
        'Incrementa tus horas de estudio al menos 2h adicionales por semana.',
        'Acude al servicio de tutoría de la UPN antes de que cierren el ciclo.',
        'Habla con tu jefe sobre flexibilidad horaria durante épocas de exámenes.',
        'Revisa si puedes reducir un curso este semestre.',
    ],
    'RIESGO_ALTO': [
        '⚡ Prioridad inmediata: habla con tu asesor académico esta semana.',
        'Evalúa retirarte de cursos en los que ya no puedas recuperar la nota.',
        'Busca el programa de becas o fraccionamiento de pensión si tienes deuda.',
        'Reorganiza tus turnos laborales para asistir al 70% mínimo de clases.',
    ],
    'DESERCION_PROBABLE': [
        '🚨 Acude AHORA al área de Bienestar Universitario de la UPN.',
        'Solicita una entrevista de retención con tu coordinador de carrera.',
        'Infórmate sobre el programa "Reserva de matrícula" para no perder tu avance.',
        'Evalúa pausar el ciclo formalmente en lugar de abandonar sin trámite.',
        'Contacta a Servicios Estudiantiles para apoyo psicológico y financiero.',
    ],
}

st.set_page_config(page_title='Predictor Académico UPN', page_icon='🎓',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.bloque-titulo{
    background:linear-gradient(90deg,#003366,#0055a4);
    color:white!important;padding:9px 18px;border-radius:8px;
    font-size:1rem;font-weight:700;margin-bottom:14px;}
.resultado-principal{
    border-radius:14px;padding:26px;margin:16px 0;color:white;
    font-size:1.5rem;font-weight:700;text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);}
.curso-badge{
    border-radius:8px;padding:10px 16px;margin:6px 0;
    color:white;font-weight:600;font-size:0.95rem;}
.semestre-box{
    border-radius:12px;padding:20px;text-align:center;
    font-size:1.3rem;font-weight:700;margin:12px 0;color:white;}
.insight-box{
    background:#f0f4ff;border-left:4px solid #003366;
    border-radius:8px;padding:14px 18px;margin:10px 0;font-size:0.95rem;}
.stButton>button{
    background-color:#003366;color:white;border-radius:8px;
    padding:12px 30px;font-size:1.05rem;font-weight:700;border:none;width:100%;}
.stButton>button:hover{background-color:#0055a4;}
h1{color:#003366!important;}
.sidebar-title{color:#003366;font-weight:700;font-size:1.1rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None, None
    modelo = joblib.load(MODEL_PATH)
    meta   = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding='utf-8') as f:
            meta = json.load(f)
    return modelo, meta


# ── HEADER ────────────────────────────────────────────────────────────
_, col_title = st.columns([1, 5])
with col_title:
    st.title('🎓 Predictor de Situación Académica')
    st.markdown('**Universidad Privada del Norte** · Modalidad para Gente que Trabaja')
st.divider()

modelo, meta = cargar_modelo()

with st.sidebar:
    st.markdown('<p class="sidebar-title">📊 Estado del Modelo</p>', unsafe_allow_html=True)
    if modelo is None:
        st.error('Modelo no entrenado aún.')
        st.info('Ve a **⚙️ Entrenar Modelo** para generarlo.')
    else:
        st.success('Modelo cargado ✓')
        if meta:
            st.metric('Accuracy (test)',         f"{meta.get('accuracy_test',0):.1%}")
            st.metric('F1-weighted',             f"{meta.get('f1_weighted',0):.1%}")
            st.metric('Registros entrenamiento', meta.get('n_registros','—'))
            fecha = meta.get('fecha_entrenamiento','')[:16].replace('T',' ')
            st.caption(f'Última actualización: {fecha}')
    st.divider()
    st.markdown('<p class="sidebar-title">📁 Dataset</p>', unsafe_allow_html=True)
    if os.path.exists(DATA_PATH):
        st.info(f"{len(pd.read_csv(DATA_PATH))} casos registrados")
    else:
        st.warning('Sin dataset aún.')

import sys
sys.path.insert(0, BASE_DIR)
from predictor_notas import (
    CURSOS_UPN, predecir_curso, predecir_semestre,
    SITUACION_CURSO_ES, SITUACION_CURSO_COLOR,
    SITUACION_SEMESTRE_ES, SITUACION_SEMESTRE_COLOR,
    RECOMENDACIONES_CURSO,
)

tab1, tab2, tab3, tab4 = st.tabs([
    '🔮 Predictor', '➕ Agregar Caso', '📊 Estadísticas', '⚙️ Entrenar Modelo'
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR UNIFICADO
# ══════════════════════════════════════════════════════════════════════
with tab1:
    if modelo is None:
        st.warning('Primero entrena el modelo en **⚙️ Entrenar Modelo**.')
        st.stop()

    st.caption('Completa los datos del estudiante y las notas de tus cursos. Al presionar **Analizar**, el modelo combinará todo para darte una predicción única e integrada.')

    # ── BLOQUE 1: DATOS GENERALES ──────────────────────────────────────
    st.markdown('<div class="bloque-titulo">📋 Datos generales del estudiante</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**📚 Académico**')
        semestre            = st.slider('Semestre actual',             1, 10,   3,          key='sem')
        cursos_matriculados = st.slider('Cursos matriculados',         1,  6,   4,          key='cmat')
        promedio_ponderado  = st.slider('Promedio ponderado (0-20)', 0.0, 20.0, 13.0, 0.5, key='prom')
        cursos_jalados      = st.slider('Cursos jalados acumulados',   0,  20,  0,          key='jal')
        turno               = st.selectbox('Turno de clases', ['noche','sabado_domingo'],   key='turno')
    with c2:
        st.markdown('**⏰ Asistencia y dedicación**')
        asistencia       = st.slider('Asistencia a clases (%)',      0, 100, 75,          key='asist')
        entrega_trabajos = st.slider('Entrega de trabajos (%)',      0, 100, 80,          key='ent')
        horas_estudio    = st.slider('Horas de estudio por semana', 0.0, 30.0, 10.0, 0.5, key='hest')
        participacion    = st.slider('Participación en clase (1-5)', 1,   5,  3,          key='part')
        uso_tutoria      = st.selectbox('¿Usa tutoría UPN?', [0,1],
                            format_func=lambda x: 'Sí' if x else 'No', key='tut')
    with c3:
        st.markdown('**💼 Personal / Laboral**')
        horas_trabajo     = st.slider('Horas de trabajo semanal',     0,  60, 40,          key='ht')
        modalidad_trabajo = st.selectbox('Modalidad de trabajo',
                            ['presencial','remoto','mixto','sin_trabajo'], key='mod')
        distancia_campus  = st.slider('Distancia al campus (km)',   0.0, 80.0, 10.0, 0.5, key='dist')
        internet_estable  = st.selectbox('¿Internet estable en casa?', [1,0],
                            format_func=lambda x: 'Sí' if x else 'No', key='inet')
        apoyo_familiar    = st.slider('Apoyo familiar (1-5)',         1,   5,  3,          key='apo')
        deuda_pension     = st.selectbox('Meses de deuda en pensión', [0,1,2,3],
                            format_func=lambda x: f'{x} mes(es)' if x else 'Al día', key='deu')

    st.divider()

    # ── BLOQUE 2: CURSOS Y NOTAS ───────────────────────────────────────
    st.markdown('<div class="bloque-titulo">📚 Notas de tus cursos — T1 / T2</div>', unsafe_allow_html=True)

    ciclos_disp = sorted({info['ciclo'] for info in CURSOS_UPN.values()})
    try:
        idx_default = ciclos_disp.index(semestre) + 1
    except ValueError:
        idx_default = 0

    ciclo_sel = st.selectbox(
        'Filtra cursos por ciclo:',
        ['Todos'] + [f'Ciclo {c}' for c in ciclos_disp],
        index=idx_default, key='ciclo_fil'
    )
    if ciclo_sel == 'Todos':
        lista_nombres = list(CURSOS_UPN.keys())
    else:
        cn = int(ciclo_sel.replace('Ciclo ',''))
        lista_nombres = [n for n, info in CURSOS_UPN.items() if info['ciclo'] == cn]

    cursos_sel = st.multiselect(
        'Selecciona tus cursos de este ciclo:',
        lista_nombres,
        default=lista_nombres[:4] if len(lista_nombres) >= 4 else lista_nombres,
        key='cursos_sel'
    )

    notas_cursos = {}
    if cursos_sel:
        for i, nombre in enumerate(cursos_sel):
            info  = CURSOS_UPN[nombre]
            emoji = {'matematica':'🔢','carrera':'💻','letras':'📖'}[info['tipo']]
            with st.expander(
                f"{emoji} **{nombre}** ({info['codigo']}) · Ciclo {info['ciclo']} · {info['creditos']} créditos",
                expanded=True
            ):
                nc1, nc2, nc3 = st.columns(3)
                with nc1:
                    t1       = st.slider('Nota T1', 0.0, 20.0, 13.0, 0.5, key=f't1_{i}')
                    tiene_t2 = st.checkbox('¿Ya tienes nota T2?', key=f'cht2_{i}')
                with nc2:
                    if tiene_t2:
                        t2 = st.slider('Nota T2', 0.0, 20.0, 12.0, 0.5, key=f't2_{i}')
                    else:
                        t2 = None
                        st.info('Solo T1 para la proyección.')
                with nc3:
                    entiende = st.slider('¿Cuánto entiendes el curso? (1=nada · 5=muy bien)',
                                         1, 5, 3, key=f'ent_{i}')
                notas_cursos[nombre] = {'t1': t1, 't2': t2, 'entiende': entiende}
    else:
        st.info('Selecciona al menos un curso para ingresar sus notas.')

    st.divider()

    # ── BOTÓN ÚNICO ────────────────────────────────────────────────────
    analizar = st.button('🔮 Analizar situación académica completa', use_container_width=True)

    if analizar:
        # ── PASO 1: Calcular resultados por curso con predictor_notas ─
        resultados_cursos = []
        for nombre, vals in notas_cursos.items():
            res = predecir_curso(
                nombre_curso    = nombre,
                t1              = vals['t1'],
                t2              = vals['t2'],
                horas_estudio   = horas_estudio,
                asistencia      = asistencia,
                entiende_curso  = vals['entiende'],
                horas_trabajo   = horas_trabajo,
                semestre_actual = semestre,
            )
            resultados_cursos.append(res)

        # ── PASO 2: Calcular las 4 features de notas reales ───────────
        if resultados_cursos:
            notas_t1_list = [r['nota_t1'] for r in resultados_cursos]
            notas_t2_list = [r['nota_t2'] for r in resultados_cursos if r['nota_t2'] is not None]
            proy_list     = [r['proyeccion_nota_final'] for r in resultados_cursos]
            creds_list    = [CURSOS_UPN[r['curso']]['creditos'] for r in resultados_cursos]

            promedio_t1_real = round(float(np.mean(notas_t1_list)), 2)
            promedio_t2_real = round(float(np.mean(notas_t2_list)), 2) if notas_t2_list else promedio_t1_real

            cursos_en_riesgo = sum(
                1 for r in resultados_cursos
                if r['situacion_curso'] in ('PROBABLE_JALE','MATEMATICAMENTE_IMPOSIBLE')
            )
            total_creditos    = sum(creds_list)
            creditos_en_riesgo= sum(
                CURSOS_UPN[r['curso']]['creditos'] for r in resultados_cursos
                if r['situacion_curso'] in ('PROBABLE_JALE','MATEMATICAMENTE_IMPOSIBLE')
            )
            pct_creditos_riesgo = round(creditos_en_riesgo / total_creditos, 2) if total_creditos > 0 else 0.0

            resumen_sem = predecir_semestre(resultados_cursos)
        else:
            # Sin cursos seleccionados: valores neutros basados en promedio histórico
            promedio_t1_real     = promedio_ponderado
            promedio_t2_real     = promedio_ponderado
            cursos_en_riesgo     = 0
            pct_creditos_riesgo  = 0.0
            resumen_sem          = None

        # ── PASO 3: Input al modelo con TODAS las features ────────────
        # Ahora el modelo recibe directamente promedio_t1, promedio_t2,
        # cursos_en_riesgo_notas y pct_creditos_riesgo como features reales
        input_ml = pd.DataFrame([{
            'semestre':                  semestre,
            'cursos_matriculados':       cursos_matriculados,
            'horas_trabajo_semanal':     horas_trabajo,
            'modalidad_trabajo':         modalidad_trabajo,
            'turno_clases':              turno,
            'asistencia_pct':            asistencia,
            'horas_estudio_semanal':     horas_estudio,
            'promedio_ponderado':        promedio_ponderado,
            'cursos_jalados_acumulados': cursos_jalados,
            'entrega_trabajos_pct':      entrega_trabajos,
            'participacion_clase':       participacion,
            'distancia_campus_km':       distancia_campus,
            'internet_estable':          internet_estable,
            'apoyo_familiar':            apoyo_familiar,
            'uso_tutoria':               uso_tutoria,
            'deuda_pension':             deuda_pension,
            # ── Features de notas reales ──
            'promedio_t1':               promedio_t1_real,
            'promedio_t2':               promedio_t2_real,
            'cursos_en_riesgo_notas':    cursos_en_riesgo,
            'pct_creditos_riesgo':       pct_creditos_riesgo,
        }])

        prediccion = modelo.predict(input_ml)[0]
        probas     = modelo.predict_proba(input_ml)[0]
        clases     = modelo.classes_

        # ══════════════════════════════════════════════════════════════
        # RESULTADO ÚNICO INTEGRADO
        # ══════════════════════════════════════════════════════════════
        st.markdown('---')
        st.markdown('## 📊 Resultado integrado')

        # Caja principal
        st.markdown(f"""
        <div class="resultado-principal" style="background:{LABEL_COLOR[prediccion]};">
            {LABEL_ES[prediccion]}
        </div>
        """, unsafe_allow_html=True)

        # Probabilidades por escenario
        st.markdown('**Probabilidad por escenario:**')
        pcols = st.columns(4)
        for i, clase in enumerate(['APROBADO_REGULAR','EN_RIESGO','RIESGO_ALTO','DESERCION_PROBABLE']):
            if clase in clases:
                idx = list(clases).index(clase)
                pcols[i].metric(LABEL_ES[clase].split(' ',1)[1], f"{probas[idx]:.1%}")

        # Insight: qué aportaron las notas a la predicción
        if resultados_cursos:
            with st.container(border=True):
                st.markdown('📌 **Variables de notas usadas en el modelo:**')
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric('Promedio T1', f"{promedio_t1_real}/20")
                ic2.metric('Promedio T2', f"{promedio_t2_real}/20")
                ic3.metric('Cursos en riesgo', f"{cursos_en_riesgo}/{len(resultados_cursos)}")
                ic4.metric('% créditos en riesgo', f"{pct_creditos_riesgo*100:.0f}%")

        # Recomendaciones generales
        st.markdown('**📋 Recomendaciones generales:**')
        for rec in RECOMENDACIONES[prediccion]:
            st.markdown(f'- {rec}')

        # Situación global del semestre
        if resumen_sem:
            st.divider()
            st.markdown('## 🎯 Situación global del semestre')
            color_sem = SITUACION_SEMESTRE_COLOR[resumen_sem['situacion_semestre']]
            etiq_sem  = SITUACION_SEMESTRE_ES[resumen_sem['situacion_semestre']]
            st.markdown(f'<div class="semestre-box" style="background:{color_sem};">{etiq_sem}</div>',
                        unsafe_allow_html=True)

            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric('✅ Probables aprobar', resumen_sem['probables_aprobacion'])
            ms2.metric('⚠️ En juego',          resumen_sem['en_juego'])
            ms3.metric('🚨 Probables jalar',   resumen_sem['probables_jale'])
            ms4.metric('⚡ Créditos en riesgo',
                       f"{resumen_sem['creditos_en_riesgo']}/{resumen_sem['creditos_totales']}")

        # Detalle por curso
        if resultados_cursos:
            st.divider()
            st.markdown('## 📚 Detalle por curso')
            for res in resultados_cursos:
                info_c  = CURSOS_UPN[res['curso']]
                emoji_c = {'matematica':'🔢','carrera':'💻','letras':'📖'}[info_c['tipo']]
                color_c = SITUACION_CURSO_COLOR[res['situacion_curso']]
                etiq_c  = SITUACION_CURSO_ES[res['situacion_curso']]
                with st.expander(f"{emoji_c} **{res['curso']}**", expanded=True):
                    st.markdown(f"""
                    <div class="curso-badge" style="background:{color_c};">
                        {etiq_c} &nbsp;|&nbsp;
                        Proyección final: <b>{res['proyeccion_nota_final']}/20</b> &nbsp;|&nbsp;
                        Nota máx. posible: <b>{res['nota_maxima_posible']}/20</b>
                    </div>
                    """, unsafe_allow_html=True)
                    if res['nota_t2'] is not None and res['puede_aprobar']:
                        min_req = res.get('nota_minima_final_requerida', 0)
                        if min_req > 20:
                            st.error(f'Necesitarías {min_req:.1f}/20 en el Final → matemáticamente imposible.')
                        elif min_req > 14:
                            st.warning(f'Necesitas al menos **{min_req:.1f}/20** en el Final.')
                        elif min_req > 0:
                            st.success(f'Necesitas al menos **{min_req:.1f}/20** en el Final para aprobar.')
                        else:
                            st.success('Ya tienes la nota asegurada.')
                    for rec in RECOMENDACIONES_CURSO[res['situacion_curso']]:
                        st.markdown(f'- {rec}')

            df_res = pd.DataFrame([{
                'Curso':       r['curso'],
                'Código':      r.get('codigo_banner','—'),
                'T1':          r['nota_t1'],
                'T2':          r['nota_t2'] if r['nota_t2'] else '—',
                'Proy. Final': r['proyeccion_nota_final'],
                'Nota Máx.':  r['nota_maxima_posible'],
                'Situación':   SITUACION_CURSO_ES[r['situacion_curso']],
            } for r in resultados_cursos])
            st.dataframe(df_res, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — AGREGAR CASO
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('➕ Agregar caso real al dataset de entrenamiento')
    st.info('Agrega estudiantes reales para mejorar la precisión del modelo.')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**📚 Académico**')
        n_semestre = st.slider('Semestre', 1, 10, 3, key='n_sem')
        n_cursos   = st.slider('Cursos matriculados', 1, 6, 4, key='n_cur')
        n_promedio = st.slider('Promedio ponderado', 0.0, 20.0, 13.0, 0.5, key='n_prom')
        n_jalados  = st.slider('Cursos jalados acumulados', 0, 20, 0, key='n_jal')
        n_turno    = st.selectbox('Turno', ['noche','sabado_domingo'], key='n_turno')
    with c2:
        st.markdown('**⏰ Dedicación**')
        n_asistencia    = st.slider('Asistencia (%)', 0, 100, 75, key='n_asist')
        n_entrega       = st.slider('Entrega trabajos (%)', 0, 100, 80, key='n_ent')
        n_hestudio      = st.slider('Horas estudio/semana', 0.0, 30.0, 10.0, 0.5, key='n_hest')
        n_participacion = st.slider('Participación (1-5)', 1, 5, 3, key='n_part')
        n_tutoria       = st.selectbox('Usa tutoría', [0,1],
                            format_func=lambda x: 'Sí' if x else 'No', key='n_tut')
    with c3:
        st.markdown('**💼 Personal/Laboral + Notas**')
        n_htrabajo   = st.slider('Horas trabajo/semana', 0, 60, 40, key='n_ht')
        n_modalidad  = st.selectbox('Modalidad trabajo',
                        ['presencial','remoto','mixto','sin_trabajo'], key='n_mod')
        n_distancia  = st.slider('Distancia campus (km)', 0.0, 80.0, 10.0, 0.5, key='n_dist')
        n_internet   = st.selectbox('Internet estable', [1,0],
                        format_func=lambda x: 'Sí' if x else 'No', key='n_int')
        n_apoyo      = st.slider('Apoyo familiar (1-5)', 1, 5, 3, key='n_apo')
        n_deuda      = st.selectbox('Meses deuda pensión', [0,1,2,3], key='n_deu')

    st.markdown('**📝 Notas del ciclo actual**')
    na1, na2, na3, na4 = st.columns(4)
    with na1:
        n_prom_t1 = st.slider('Promedio T1', 0.0, 20.0, 13.0, 0.5, key='n_pt1')
    with na2:
        n_prom_t2 = st.slider('Promedio T2', 0.0, 20.0, 12.0, 0.5, key='n_pt2')
    with na3:
        n_cursos_riesgo = st.slider('Cursos con riesgo de jale', 0, 6, 0, key='n_crisk')
    with na4:
        n_pct_riesgo = st.slider('% créditos en riesgo', 0.0, 1.0, 0.0, 0.05, key='n_pctrisk')

    n_situacion = st.selectbox(
        '📌 Situación académica real del estudiante',
        ['APROBADO_REGULAR','EN_RIESGO','RIESGO_ALTO','DESERCION_PROBABLE'],
        format_func=lambda x: LABEL_ES[x]
    )

    if st.button('💾 Guardar caso al dataset', use_container_width=True):
        nuevo = {
            'semestre': n_semestre, 'cursos_matriculados': n_cursos,
            'horas_trabajo_semanal': n_htrabajo, 'modalidad_trabajo': n_modalidad,
            'turno_clases': n_turno, 'asistencia_pct': n_asistencia,
            'horas_estudio_semanal': n_hestudio, 'promedio_ponderado': n_promedio,
            'cursos_jalados_acumulados': n_jalados, 'entrega_trabajos_pct': n_entrega,
            'participacion_clase': n_participacion, 'distancia_campus_km': n_distancia,
            'internet_estable': n_internet, 'apoyo_familiar': n_apoyo,
            'uso_tutoria': n_tutoria, 'deuda_pension': n_deuda,
            'promedio_t1': n_prom_t1, 'promedio_t2': n_prom_t2,
            'cursos_en_riesgo_notas': n_cursos_riesgo,
            'pct_creditos_riesgo': n_pct_riesgo,
            'situacion_academica': n_situacion,
        }
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df_exist = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
        df_final = pd.concat([df_exist, pd.DataFrame([nuevo])], ignore_index=True)
        df_final.to_csv(DATA_PATH, index=False)
        st.success(f'✅ Caso guardado. Total: {len(df_final)} registros.')


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('📊 Estadísticas del dataset de entrenamiento')
    if not os.path.exists(DATA_PATH):
        st.warning('No hay dataset aún.')
    else:
        df = pd.read_csv(DATA_PATH)
        st.metric('Total de registros', len(df))
        dist = df['situacion_academica'].value_counts().reset_index()
        dist.columns = ['Situación','Cantidad']
        dist['Situación_ES'] = dist['Situación'].map(LABEL_ES)
        try:
            import plotly.express as px
            fig = px.pie(dist, values='Cantidad', names='Situación_ES',
                         color='Situación',
                         color_discrete_map={k:v for k,v in LABEL_COLOR.items()},
                         hole=0.35)
            fig.update_layout(margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(dist)

        cols_resumen = ['promedio_ponderado','asistencia_pct','horas_trabajo_semanal']
        for col in ['promedio_t1','promedio_t2']:
            if col in df.columns:
                cols_resumen.append(col)
        resumen = df.groupby('situacion_academica')[cols_resumen].mean().round(2)
        resumen.index = resumen.index.map(LABEL_ES)
        st.dataframe(resumen, use_container_width=True)
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button('⬇️ Descargar dataset',
                           df.to_csv(index=False).encode('utf-8'),
                           'estudiantes_upn.csv','text/csv')


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — ENTRENAR MODELO
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader('⚙️ Entrenamiento del modelo predictivo')
    st.info('⚠️ Si actualizaste el sistema, genera un dataset nuevo y re-entrena para que el modelo incluya las features de notas T1/T2.')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('**Paso 1: Generar / actualizar dataset**')
        n_generar = st.number_input('Casos sintéticos a generar',
                                    min_value=100, max_value=5000, value=1200, step=100)
        if st.button('🔄 Generar dataset sintético', use_container_width=True):
            with st.spinner('Generando datos con features de notas T1/T2...'):
                from generar_dataset import generar_dataset as gen_ds
                df_gen = gen_ds(n_generar)
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                # Si el CSV viejo existe pero no tiene las nuevas columnas, lo reemplazamos
                if os.path.exists(DATA_PATH):
                    df_exist = pd.read_csv(DATA_PATH)
                    if 'promedio_t1' in df_exist.columns:
                        df_final = pd.concat([df_exist, df_gen], ignore_index=True)
                    else:
                        df_final = df_gen  # Dataset viejo incompatible → reemplazar
                        st.warning('Dataset anterior sin features de notas → reemplazado con datos nuevos.')
                else:
                    df_final = df_gen
                df_final.to_csv(DATA_PATH, index=False)
            st.success(f'✅ Dataset actualizado: {len(df_final)} registros totales.')

    with col_b:
        st.markdown('**Paso 2: Entrenar el modelo**')
        st.caption('Random Forest con features de notas T1/T2 integradas.')
        if st.button('🚀 Entrenar modelo ahora', use_container_width=True):
            if not os.path.exists(DATA_PATH):
                st.error('Primero genera el dataset (Paso 1).')
            else:
                with st.spinner('Entrenando... puede tomar 30-60 seg.'):
                    import importlib, entrenar_modelo as em
                    importlib.reload(em)
                    pipeline, meta_result = em.entrenar()
                st.success('✅ Modelo entrenado y guardado.')
                st.metric('Accuracy',      f"{meta_result['accuracy_test']:.1%}")
                st.metric('F1-weighted',   f"{meta_result['f1_weighted']:.1%}")
                st.metric('CV-5 Accuracy', f"{meta_result['cv_accuracy_mean']:.1%} ± {meta_result['cv_accuracy_std']:.1%}")
                st.info('Recarga la página (F5) para activar el modelo nuevo.')

    st.divider()
    st.markdown("""
    **📖 Flujo para activar el sistema integrado:**
    1. Genera el dataset nuevo (ya incluye `promedio_t1`, `promedio_t2`, `cursos_en_riesgo_notas`, `pct_creditos_riesgo`).
    2. Entrena el modelo con esas features.
    3. En **Predictor**, ingresa datos generales + notas T1/T2 → el modelo usa **todo junto** para predecir.
    """)

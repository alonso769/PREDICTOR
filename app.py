# -*- coding: utf-8 -*-
"""
app.py
-------
Dashboard interactivo (Streamlit) del Sistema Predictivo de Rendimiento
Académico para estudiantes universitarios que trabajan.

Estructura (pestañas):
  1. Predicción Individual  -> formulario + lógica T1/T2/Nota Final +
     regresión lineal (si no tiene nota final) + clasificación
     (Random Forest / Regresión Logística / Red Neuronal) + cluster.
  2. Comparación de Modelos -> métricas de clasificación y regresión.
  3. Perfiles Académicos (Clustering) -> visualización K-Means + PCA.
  4. Ética en la IA -> mitigación de sesgos y disclaimer.
  5. Exploración del Dataset -> vistazo general a los datos sintéticos.

Ejecutar con:  streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

MODELS_DIR = "models"
DATA_DIR = "data"

st.set_page_config(
    page_title="Sistema Predictivo de Rendimiento Académico",
    page_icon="🎓",
    layout="wide",
)

# ----------------------------------------------------------------------
# Utilidades de carga (con cache para no releer disco en cada interacción)
# ----------------------------------------------------------------------

@st.cache_resource
def cargar_artefactos():
    faltantes = []
    archivos = {
        "reg_lineal": "regresion_lineal_examen.pkl",
        "scaler_reg": "scaler_regresion.pkl",
        "log_reg": "logistic_regression.pkl",
        "rf": "random_forest.pkl",
        "scaler_clf": "scaler_clasificacion.pkl",
        "mlp": "mlp_neural_network.pkl",
        "kmeans": "kmeans.pkl",
        "scaler_cluster": "scaler_clustering.pkl",
        "pca": "pca_clustering.pkl",
        "etiquetas_clusters": "etiquetas_clusters.pkl",
        "encoder_modalidad": "encoder_modalidad.pkl",
        "encoder_situacion": "encoder_situacion.pkl",
    }
    objetos = {}
    for clave, archivo in archivos.items():
        ruta = os.path.join(MODELS_DIR, archivo)
        if not os.path.exists(ruta):
            faltantes.append(archivo)
        else:
            objetos[clave] = joblib.load(ruta)

    if faltantes:
        return None, faltantes

    with open(os.path.join(MODELS_DIR, "metricas.json"), "r", encoding="utf-8") as f:
        objetos["metricas"] = json.load(f)

    return objetos, []


@st.cache_data
def cargar_dataset():
    ruta = os.path.join(DATA_DIR, "dataset_con_clusters.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    ruta_base = os.path.join(DATA_DIR, "dataset_estudiantes.csv")
    if os.path.exists(ruta_base):
        return pd.read_csv(ruta_base)
    return None


ART, FALTANTES = cargar_artefactos()
DF = cargar_dataset()

if ART is None:
    st.error(
        "⚠️ No se encontraron los artefactos entrenados: "
        + ", ".join(FALTANTES)
        + ".\n\nEjecuta primero `python entrenamiento.py` en esta misma carpeta "
        "para generar los datos y entrenar los modelos antes de iniciar el dashboard."
    )
    st.stop()

FEATURES_REG = ART["metricas"]["features_regresion_examen"]
FEATURES_CLF = ART["metricas"]["features_clasificacion"]
FEATURES_CLUSTER = ART["metricas"]["features_clustering"]
CLASES_SITUACION = list(ART["encoder_situacion"].classes_)

COLOR_SITUACION = {
    "Aprobado": "#2ecc71",
    "Riesgo": "#f39c12",
    "Desercion": "#e74c3c",
}

# ----------------------------------------------------------------------
# Sidebar de navegación
# ----------------------------------------------------------------------
st.sidebar.title("🎓 Sistema Predictivo")
st.sidebar.caption("Rendimiento académico de estudiantes que trabajan")
pagina = st.sidebar.radio(
    "Navegación",
    [
        "🔮 Predicción Individual",
        "📊 Comparación de Modelos",
        "🧩 Perfiles Académicos (Clustering)",
        "⚖️ Ética en la IA",
        "🗃️ Exploración del Dataset",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "Proyecto final — Curso de Sistemas Inteligentes.\n\n"
    "Modelos: Regresión Lineal, Regresión Logística, Random Forest, "
    "Red Neuronal (MLP) y K-Means."
)


# ========================================================================
# PÁGINA 1: PREDICCIÓN INDIVIDUAL
# ========================================================================
def pagina_prediccion():
    st.title("🔮 Predicción de Rendimiento Académico")
    st.markdown(
        "Completa los datos del estudiante. Si aún no tienes la **Nota Final**, "
        "el sistema la **predice** con un modelo de Regresión Lineal a partir "
        "de tus hábitos."
    )

    col_izq, col_der = st.columns([1, 1])

    # ------------------ Columna izquierda: notas ------------------
    with col_izq:
        st.subheader("1️⃣ Evaluaciones")
        T1 = st.slider("Nota T1 (0 - 20)", 0.0, 20.0, 12.0, 0.5)
        T2 = st.slider("Nota T2 (0 - 20)", 0.0, 20.0, 12.0, 0.5)

        tiene_nota_final = st.radio(
            "¿Ya tienes tu Nota Final (examen final ya rendido)?",
            ["No, todavía no la tengo", "Sí, ya la tengo"],
            index=0,
        )

    # ------------------ Columna derecha: hábitos -------------------
    with col_der:
        st.subheader("2️⃣ Hábitos y contexto")
        horas_trabajo = st.slider("Horas de trabajo semanales", 0, 60, 25)
        horas_estudio = st.slider("Horas de estudio semanales", 0, 40, 12)
        asistencia = st.slider("Asistencia a clases (%)", 0, 100, 80)
        entendimiento = st.slider("Entendimiento del curso (1-10)", 1, 10, 6)
        horas_sueno = st.slider("Horas de sueño diarias", 3.0, 10.0, 6.5, 0.5)
        nivel_estres = st.slider("Nivel de estrés autopercibido (1-10)", 1, 10, 5)
        carga_familiar = st.slider("Personas que dependen económicamente de ti", 0, 5, 0)

    with st.expander("➕ Datos adicionales (mejoran la precisión del perfil / cluster)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            edad = st.number_input("Edad", 17, 45, 22)
            ciclo = st.number_input("Ciclo académico", 1, 12, 5)
        with c2:
            tiene_beca = st.selectbox("¿Tiene beca?", ["No", "Sí"]) == "Sí"
            num_cursos = st.number_input("Cursos matriculados este ciclo", 1, 10, 5)
        with c3:
            modalidad = st.selectbox(
                "Modalidad de trabajo",
                list(ART["encoder_modalidad"].classes_),
            )
            tiempo_transporte = st.slider("Horas diarias en transporte", 0.0, 4.0, 1.0, 0.1)
        salud_mental = st.slider(
            "Salud mental autopercibida (1-10)", 1, 10,
            max(1, 10 - nivel_estres // 1) if nivel_estres else 6,
        )

    st.divider()

    # --- Widget condicional de nota final manual ---
    nota_final_manual = None
    if tiene_nota_final == "Sí, ya la tengo":
        nota_final_manual = st.number_input(
            "Ingresa tu Nota Final (0-20)", 0.0, 20.0, 13.0, 0.5, key="nota_final_input"
        )

    if st.button("📈 Calcular resultado", type="primary", use_container_width=True):
        entrada_reg = pd.DataFrame([{
            "horas_estudio_semanal": horas_estudio,
            "horas_trabajo_semanal": horas_trabajo,
            "asistencia_pct": asistencia,
            "entendimiento_curso": entendimiento,
            "horas_sueno": horas_sueno,
            "nivel_estres": nivel_estres,
            "carga_familiar": carga_familiar,
            "T1": T1,
            "T2": T2,
        }])[FEATURES_REG]

        if tiene_nota_final == "Sí, ya la tengo":
            nota_final = float(nota_final_manual)
            examen_final_usado = round((nota_final - 0.20 * T1 - 0.30 * T2) / 0.50, 2)
            origen_nota = "Ingresada manualmente por el usuario"
        else:
            X_reg_s = ART["scaler_reg"].transform(entrada_reg)
            examen_final_usado = float(ART["reg_lineal"].predict(X_reg_s)[0])
            examen_final_usado = round(np.clip(examen_final_usado, 0, 20), 2)
            nota_final = round(0.20 * T1 + 0.30 * T2 + 0.50 * examen_final_usado, 2)
            origen_nota = "Predicha con Regresión Lineal (Machine Learning)"

        st.session_state["resultado"] = {
            "nota_final": nota_final,
            "examen_final_usado": examen_final_usado,
            "origen_nota": origen_nota,
            "T1": T1, "T2": T2,
            "horas_trabajo": horas_trabajo, "horas_estudio": horas_estudio,
            "asistencia": asistencia, "entendimiento": entendimiento,
            "horas_sueno": horas_sueno, "nivel_estres": nivel_estres,
            "carga_familiar": carga_familiar, "edad": edad, "ciclo": ciclo,
            "tiene_beca": int(tiene_beca), "num_cursos": num_cursos,
            "modalidad": modalidad, "tiempo_transporte": tiempo_transporte,
            "salud_mental": salud_mental,
        }

    if "resultado" not in st.session_state:
        st.info("Completa el formulario y presiona **Calcular resultado**.")
        return

    r = st.session_state["resultado"]

    st.header("📋 Resultado")
    m1, m2, m3 = st.columns(3)
    m1.metric("Nota Final estimada", f"{r['nota_final']:.2f} / 20")
    m2.metric("Examen Final usado", f"{r['examen_final_usado']:.2f} / 20")
    m3.metric("Origen del dato", "🧮 ML" if "Predicha" in r["origen_nota"] else "✍️ Manual")
    st.caption(r["origen_nota"])

    # ---------------- 2. Clasificación de Situación Académica ----------------
    st.subheader("🎯 Situación Académica Predicha")

    entrada_clf = pd.DataFrame([{
        "edad": r["edad"], "ciclo": r["ciclo"],
        "horas_trabajo_semanal": r["horas_trabajo"],
        "tiempo_transporte": r["tiempo_transporte"],
        "carga_familiar": r["carga_familiar"], "tiene_beca": r["tiene_beca"],
        "num_cursos_matriculados": r["num_cursos"],
        "horas_estudio_semanal": r["horas_estudio"], "horas_sueno": r["horas_sueno"],
        "nivel_estres": r["nivel_estres"], "salud_mental_autopercibida": r["salud_mental"],
        "asistencia_pct": r["asistencia"], "entendimiento_curso": r["entendimiento"],
        "T1": r["T1"], "T2": r["T2"],
        "modalidad_trabajo_enc": ART["encoder_modalidad"].transform([r["modalidad"]])[0],
    }])[FEATURES_CLF]

    X_clf_s = ART["scaler_clf"].transform(entrada_clf)

    modelos_clf = {
        "Random Forest (principal)": ART["rf"],
        "Regresión Logística (base)": ART["log_reg"],
        "Red Neuronal (MLP)": ART["mlp"],
    }

    tabs_modelos = st.tabs(list(modelos_clf.keys()))
    pred_principal = None
    for tab, (nombre, modelo) in zip(tabs_modelos, modelos_clf.items()):
        with tab:
            pred = modelo.predict(X_clf_s)[0]
            etiqueta_pred = ART["encoder_situacion"].inverse_transform([pred])[0]
            proba = modelo.predict_proba(X_clf_s)[0]
            clases = ART["encoder_situacion"].classes_

            if nombre.startswith("Random Forest"):
                pred_principal = etiqueta_pred

            color = COLOR_SITUACION.get(etiqueta_pred, "#3498db")
            st.markdown(
                f"### Predicción: <span style='color:{color}'>**{etiqueta_pred}**</span>",
                unsafe_allow_html=True,
            )
            df_proba = pd.DataFrame({"Situación": clases, "Probabilidad": proba})
            fig = px.bar(
                df_proba, x="Situación", y="Probabilidad", color="Situación",
                color_discrete_map=COLOR_SITUACION, text_auto=".1%",
                range_y=[0, 1],
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- 3. Explicación lógica (transparencia) ----------------
    with st.expander("🔍 ¿Por qué esta predicción? (factores clave)"):
        st.markdown(f"""
- **Balance carga académica vs. laboral:** {r['horas_estudio']}h de estudio vs
  {r['horas_trabajo']}h de trabajo semanales.
- **Asistencia:** {r['asistencia']}% (referencia de aprobación ≥ 55%).
- **Nivel de estrés:** {r['nivel_estres']}/10 — variable con más peso en el
  Random Forest para distinguir "Riesgo" de "Deserción".
- **Carga familiar:** {r['carga_familiar']} persona(s) dependientes — impacta
  el tiempo disponible para estudiar, no la capacidad del estudiante.
        """)
        st.caption(
            "Nota: el modelo NO usa 'trabaja o no trabaja' como variable "
            "determinante aislada — ver pestaña **Ética en la IA**."
        )

    # ---------------- 4. Perfil académico (cluster) ----------------
    st.subheader("🧩 Perfil Académico (Clustering K-Means)")
    entrada_cluster = pd.DataFrame([{
        "horas_trabajo_semanal": r["horas_trabajo"],
        "horas_estudio_semanal": r["horas_estudio"],
        "asistencia_pct": r["asistencia"],
        "nivel_estres": r["nivel_estres"],
        "entendimiento_curso": r["entendimiento"],
        "carga_familiar": r["carga_familiar"],
    }])[FEATURES_CLUSTER]

    X_cluster_s = ART["scaler_cluster"].transform(entrada_cluster)
    cluster_id = int(ART["kmeans"].predict(X_cluster_s)[0])
    etiqueta_cluster = ART["etiquetas_clusters"].get(str(cluster_id), ART["etiquetas_clusters"].get(cluster_id, "N/D"))

    st.success(f"Perfil asignado: **{etiqueta_cluster}**")

    perfiles_desc = {
        "Trabajador Esforzado": "Equilibra bien trabajo y estudio; mantiene buena asistencia y bajo estrés relativo.",
        "Riesgo por Ausentismo Laboral": "El trabajo reduce fuertemente su asistencia y tiempo de estudio.",
        "Estudiante de Alto Rendimiento": "Dedica poco tiempo a trabajar y mucho a estudiar; alto rendimiento esperado.",
        "Vulnerable por Carga Familiar y Estres": "Combina trabajo, responsabilidades familiares y estrés elevado: requiere apoyo institucional.",
        "Perfil Mixto / Regular": "No presenta un patrón extremo en ninguna dimensión.",
    }
    st.caption(perfiles_desc.get(etiqueta_cluster, ""))

    if pred_principal in ("Riesgo", "Desercion"):
        st.warning(
            "⚠️ Este resultado es una **estimación estadística**, no un diagnóstico. "
            "Se recomienda acercarse a Tutoría Académica / Bienestar Universitario "
            "para recibir acompañamiento personalizado."
        )


# ========================================================================
# PÁGINA 2: COMPARACIÓN DE MODELOS
# ========================================================================
def pagina_comparacion():
    st.title("📊 Comparación de Modelos")
    metricas = ART["metricas"]

    st.subheader("Regresión Lineal — Predicción del Examen Final")
    reg = metricas["regresion_lineal_examen_final"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{reg['r2']:.3f}")
    c2.metric("MAE", f"{reg['mae']:.3f}")
    c3.metric("RMSE", f"{reg['rmse']:.3f}")

    coefs = pd.DataFrame(
        list(reg["coeficientes"].items()), columns=["Variable", "Coeficiente"]
    ).sort_values("Coeficiente")
    fig_coef = px.bar(
        coefs, x="Coeficiente", y="Variable", orientation="h",
        color="Coeficiente", color_continuous_scale="RdYlGn",
        title="Impacto de cada variable en la nota del Examen Final",
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()
    st.subheader("Clasificación de Situación Académica")
    clf = metricas["clasificacion"]

    tabla = pd.DataFrame({
        "Modelo": ["Regresión Logística (base)", "Random Forest (principal)"],
        "Accuracy": [clf["logistic_regression"]["accuracy"], clf["random_forest"]["accuracy"]],
        "F1 (macro)": [clf["logistic_regression"]["f1_macro"], clf["random_forest"]["f1_macro"]],
        "Precisión (macro)": [clf["logistic_regression"]["precision_macro"], clf["random_forest"]["precision_macro"]],
        "Recall (macro)": [clf["logistic_regression"]["recall_macro"], clf["random_forest"]["recall_macro"]],
    })
    st.dataframe(tabla.style.format({c: "{:.3f}" for c in tabla.columns[1:]}), use_container_width=True)

    fig_comp = go.Figure()
    for _, fila in tabla.iterrows():
        fig_comp.add_trace(go.Bar(
            name=fila["Modelo"],
            x=["Accuracy", "F1 (macro)", "Precisión", "Recall"],
            y=[fila["Accuracy"], fila["F1 (macro)"], fila["Precisión (macro)"], fila["Recall (macro)"]],
        ))
    fig_comp.update_layout(barmode="group", title="Regresión Logística vs Random Forest")
    st.plotly_chart(fig_comp, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cm_lr = np.array(clf["logistic_regression"]["confusion_matrix"])
        fig_cm_lr = px.imshow(
            cm_lr, text_auto=True, x=CLASES_SITUACION, y=CLASES_SITUACION,
            labels=dict(x="Predicho", y="Real"), title="Matriz de Confusión — Reg. Logística",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm_lr, use_container_width=True)
    with col_b:
        cm_rf = np.array(clf["random_forest"]["confusion_matrix"])
        fig_cm_rf = px.imshow(
            cm_rf, text_auto=True, x=CLASES_SITUACION, y=CLASES_SITUACION,
            labels=dict(x="Predicho", y="Real"), title="Matriz de Confusión — Random Forest",
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)

    importancias = clf["random_forest"].get("feature_importance", {})
    if importancias:
        df_imp = pd.DataFrame(list(importancias.items()), columns=["Variable", "Importancia"])
        df_imp = df_imp.sort_values("Importancia", ascending=True).tail(10)
        fig_imp = px.bar(df_imp, x="Importancia", y="Variable", orientation="h",
                          title="Top 10 variables más influyentes (Random Forest)")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("🧠 Red Neuronal (MLP) + búsqueda de hiperparámetros")
    rn = metricas["red_neuronal"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**MLP Base (sin optimizar)**")
        st.metric("Accuracy", f"{rn['mlp_base']['accuracy']:.3f}")
        st.metric("F1 (macro)", f"{rn['mlp_base']['f1_macro']:.3f}")
    with c2:
        st.markdown("**MLP + GridSearchCV**")
        st.metric("Accuracy", f"{rn['mlp_optimizado_gridsearch']['accuracy']:.3f}")
        st.metric("F1 (macro)", f"{rn['mlp_optimizado_gridsearch']['f1_macro']:.3f}")

    st.info(
        f"✅ **Modelo final seleccionado:** MLP {rn['modelo_final_seleccionado']} "
        f"(criterio: mejor F1-macro en test). "
        f"Mejores hiperparámetros probados vía GridSearchCV: "
        f"`{rn['mlp_optimizado_gridsearch']['mejores_hiperparametros']}` "
        f"(F1-macro promedio en validación cruzada: "
        f"{rn['mlp_optimizado_gridsearch']['cv_best_score_f1_macro']:.3f})."
    )
    st.caption(
        "La búsqueda de hiperparámetros se evalúa con validación cruzada "
        "(no directamente sobre el set de prueba) para evitar sobreajuste "
        "en la selección del modelo. El sistema conserva siempre el modelo "
        "con mejor desempeño real, sea el base o el optimizado."
    )


# ========================================================================
# PÁGINA 3: CLUSTERING
# ========================================================================
def pagina_clustering():
    st.title("🧩 Perfiles Académicos (Aprendizaje No Supervisado)")
    st.markdown(
        "Se aplicó **K-Means** sobre variables de comportamiento (horas de "
        "trabajo, horas de estudio, asistencia, estrés, entendimiento y carga "
        "familiar) para agrupar a los estudiantes en perfiles, sin usar la "
        "nota como variable de entrada."
    )

    if DF is None or "cluster" not in DF.columns:
        st.warning("No se encontró el dataset con clusters. Ejecuta `entrenamiento.py`.")
        return

    etiquetas = ART["etiquetas_clusters"]
    conteo = DF["perfil_academico"].value_counts().reset_index()
    conteo.columns = ["Perfil", "Cantidad"]

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_pie = px.pie(conteo, names="Perfil", values="Cantidad",
                          title="Distribución de estudiantes por perfil")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(
            DF, x="pca_x", y="pca_y", color="perfil_academico",
            title="Proyección PCA (2D) de los clusters",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Características promedio por perfil")
    cols_resumen = ["horas_trabajo_semanal", "horas_estudio_semanal", "asistencia_pct",
                     "nivel_estres", "entendimiento_curso", "carga_familiar", "nota_final"]
    resumen = DF.groupby("perfil_academico")[cols_resumen].mean().round(2)
    st.dataframe(resumen, use_container_width=True)

    st.subheader("Situación académica dentro de cada perfil")
    tabla_situacion = pd.crosstab(DF["perfil_academico"], DF["situacion_academica"], normalize="index") * 100
    fig_stack = px.bar(
        tabla_situacion, barmode="stack",
        title="% de situación académica por perfil",
        color_discrete_map=COLOR_SITUACION,
        labels={"value": "% de estudiantes", "perfil_academico": "Perfil"},
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    st.subheader("Descripción de los perfiles detectados")
    perfiles_desc = {
        "Trabajador Esforzado": "🟢 Combina trabajo y estudio de forma equilibrada. Buena asistencia, bajo estrés relativo. Mayoría 'Aprobado'.",
        "Riesgo por Ausentismo Laboral": "🟠 El trabajo compite fuertemente con la asistencia y el estudio. Requiere flexibilidad horaria.",
        "Estudiante de Alto Rendimiento": "🟢 Poca o nula carga laboral, mucho tiempo de estudio. Rendimiento sobresaliente.",
        "Vulnerable por Carga Familiar y Estres": "🔴 Trabaja, tiene dependientes económicos y estrés elevado. Perfil prioritario para apoyo institucional (becas, tutoría, salud mental).",
        "Perfil Mixto / Regular": "🟡 No presenta un patrón extremo; comportamiento cercano al promedio general.",
    }
    for nombre in conteo["Perfil"]:
        base_nombre = nombre.split(" (")[0]
        st.markdown(f"**{nombre}**: {perfiles_desc.get(base_nombre, 'Perfil sin descripción predefinida.')}")


# ========================================================================
# PÁGINA 4: ÉTICA EN LA IA
# ========================================================================
def pagina_etica():
    st.title("⚖️ Ética en la Inteligencia Artificial")

    st.markdown("""
Este sistema fue diseñado teniendo en cuenta que **el estudiante que trabaja
no debe ser penalizado por el hecho de trabajar**, sino evaluado por el
**efecto real y medible** de sus circunstancias sobre su desempeño.
""")

    st.subheader("1. Mitigación de sesgos en los datos")
    st.markdown("""
- **No se usa "trabaja: sí/no" como variable determinante aislada.** El
  dataset incluye un grupo de contraste de estudiantes que no trabajan,
  precisamente para que los modelos aprendan que el problema no es trabajar
  en sí, sino la *falta de balance* entre horas de trabajo, horas de estudio,
  descanso y apoyo familiar.
- **Variables de esfuerzo y contexto, no solo de resultado.** Se incorporan
  `horas_estudio_semanal`, `entendimiento_curso`, `asistencia_pct` y
  `carga_familiar` junto a `horas_trabajo_semanal`, de modo que un estudiante
  que trabaja muchas horas pero mantiene buena asistencia y estudia lo
  suficiente puede seguir siendo clasificado como "Aprobado" — así se refleja
  en el perfil **"Trabajador Esforzado"** del módulo de clustering.
- **Balanceo de clases (SMOTE).** La clase "Deserción" es naturalmente
  minoritaria (~2% del dataset). Sin balanceo, un modelo perezoso podría
  ignorarla por completo y aun así lograr alta *accuracy* general. SMOTE
  genera ejemplos sintéticos de la clase minoritaria en el set de
  entrenamiento para que el modelo le preste atención real, sin tocar el
  set de prueba (evitando fuga de datos / *data leakage*).
- **Random Forest con `class_weight="balanced"`** como capa adicional de
  mitigación, y comparación explícita contra Regresión Logística para
  detectar si un modelo más complejo introduce sesgos que el modelo simple
  no tiene.
- **Manejo de outliers sin eliminar estudiantes.** Se usa recorte por rango
  intercuartílico (IQR) en vez de descartar filas, para no borrar del
  dataset a los casos extremos reales (ej. alguien que trabaja 90h/semana),
  que suelen ser justamente los estudiantes más vulnerables.
""")

    st.subheader("2. Transparencia del modelo")
    st.markdown("""
- En la pestaña de predicción se muestra el **razonamiento detrás de cada
  resultado** (factores clave) y la **probabilidad** de cada clase, no solo
  la etiqueta final.
- Se reportan honestamente los casos en que el modelo optimizado (MLP +
  GridSearchCV) **no** superó al modelo base en el conjunto de prueba, en
  vez de ocultar ese resultado.
""")

    st.subheader("3. Límites del sistema")
    st.warning("""
**Este sistema es una herramienta de apoyo, NO un diagnóstico ni una decisión
administrativa.** Las predicciones se basan en patrones estadísticos de un
dataset sintético construido para fines académicos y **no deben usarse**
para:
- Tomar decisiones de matrícula, beca o expulsión de un estudiante real.
- Etiquetar a una persona real sin su conocimiento y consentimiento.
- Sustituir la evaluación de un tutor académico, psicólogo o consejero.

Toda predicción de "Riesgo" o "Deserción" debe entenderse como una **señal
de alerta temprana para ofrecer apoyo**, nunca como una sentencia.
""")

    st.subheader("4. Privacidad de los datos")
    st.markdown("""
El dataset utilizado es **100% sintético**, generado con `generador_datos.py`
mediante distribuciones estadísticas y correlaciones definidas por el equipo
desarrollador. No contiene información real de estudiantes. Si este sistema
se adaptara a datos reales, sería indispensable anonimizar identificadores,
obtener consentimiento informado y cumplir con la normativa de protección
de datos personales vigente.
""")


# ========================================================================
# PÁGINA 5: EXPLORACIÓN DEL DATASET
# ========================================================================
def pagina_dataset():
    st.title("🗃️ Exploración del Dataset Sintético")

    if DF is None:
        st.warning("No se encontró ningún dataset. Ejecuta `generador_datos.py` o `entrenamiento.py`.")
        return

    st.markdown(f"**Filas:** {len(DF):,}  |  **Columnas:** {DF.shape[1]}")
    st.dataframe(DF.head(50), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_dist = px.histogram(
            DF, x="nota_final", color="situacion_academica",
            color_discrete_map=COLOR_SITUACION, nbins=30,
            title="Distribución de Nota Final por Situación Académica",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(
            DF.sample(min(1500, len(DF)), random_state=42),
            x="horas_trabajo_semanal", y="nota_final",
            color="situacion_academica", color_discrete_map=COLOR_SITUACION,
            title="Horas de trabajo vs. Nota Final",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Matriz de correlación (variables numéricas)")
    cols_num = DF.select_dtypes(include=[np.number]).columns.tolist()
    cols_num = [c for c in cols_num if c not in ("pca_x", "pca_y", "cluster")]
    corr = DF[cols_num].corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Correlación entre variables")
    st.plotly_chart(fig_corr, use_container_width=True)


# ========================================================================
# ROUTER
# ========================================================================
if pagina == "🔮 Predicción Individual":
    pagina_prediccion()
elif pagina == "📊 Comparación de Modelos":
    pagina_comparacion()
elif pagina == "🧩 Perfiles Académicos (Clustering)":
    pagina_clustering()
elif pagina == "⚖️ Ética en la IA":
    pagina_etica()
elif pagina == "🗃️ Exploración del Dataset":
    pagina_dataset()

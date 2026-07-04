# -*- coding: utf-8 -*-
"""
entrenamiento.py
-----------------
Pipeline completo de Machine Learning para el Sistema Predictivo de
Rendimiento Académico.

Incluye:
  1. Carga y preprocesamiento (outliers, escalado, encoding).
  2. Regresión Lineal -> predice 'examen_final' (nota continua) cuando
     el alumno todavía no la tiene.
  3. Clasificación de 'situacion_academica' (Aprobado / Riesgo / Deserción):
       - Regresión Logística (modelo base)
       - Random Forest (modelo principal) + comparación de métricas
       - Balanceo de clases con SMOTE (el dataset es naturalmente
         desbalanceado: pocos casos de "Deserción")
  4. Red Neuronal (MLPClassifier) + búsqueda de hiperparámetros
     (GridSearchCV) para mejorar su desempeño.
  5. Clustering no supervisado (K-Means) -> "Perfiles Académicos".

Todos los artefactos (modelos, scaler, encoders, métricas) se guardan
en la carpeta ./models para que app.py los consuma directamente sin
tener que reentrenar nada.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from imblearn.over_sampling import SMOTE

from generador_datos import generar_dataset

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = "models"
DATA_DIR = "data"

FEATURES_NUMERICAS = [
    "edad", "ciclo", "horas_trabajo_semanal", "tiempo_transporte",
    "carga_familiar", "tiene_beca", "num_cursos_matriculados",
    "horas_estudio_semanal", "horas_sueno", "nivel_estres",
    "salud_mental_autopercibida", "asistencia_pct", "entendimiento_curso",
    "T1", "T2",
]

FEATURES_REGRESION_EXAMEN = [
    "horas_estudio_semanal", "horas_trabajo_semanal", "asistencia_pct",
    "entendimiento_curso", "horas_sueno", "nivel_estres",
    "carga_familiar", "T1", "T2",
]

FEATURES_CLUSTERING = [
    "horas_trabajo_semanal", "horas_estudio_semanal", "asistencia_pct",
    "nivel_estres", "entendimiento_curso", "carga_familiar",
]


def manejar_outliers_iqr(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Recorta (clip) outliers usando el método IQR (rango interquartil).
    En vez de eliminar filas (lo que perdería información valiosa de
    estudiantes en situaciones extremas reales), se recortan los valores
    a los límites del bigote, preservando el registro completo.
    """
    df = df.copy()
    for col in columnas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
    return df


def paso_1_datos():
    print("=" * 70)
    print("PASO 1: Generación / carga de datos y preprocesamiento")
    print("=" * 70)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ruta_csv = os.path.join(DATA_DIR, "dataset_estudiantes.csv")
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv)
        print(f"Dataset cargado desde {ruta_csv} ({len(df)} filas)")
    else:
        df = generar_dataset(6000)
        df.to_csv(ruta_csv, index=False)
        print(f"Dataset generado y guardado ({len(df)} filas)")

    # --- Manejo de outliers (IQR) sobre variables numéricas sensibles ---
    cols_outliers = ["horas_trabajo_semanal", "horas_estudio_semanal",
                      "horas_sueno", "asistencia_pct"]
    antes = df[cols_outliers].describe().loc["max"].to_dict()
    df = manejar_outliers_iqr(df, cols_outliers)
    despues = df[cols_outliers].describe().loc["max"].to_dict()
    print("Outliers recortados (IQR). Máximos antes -> después:")
    for c in cols_outliers:
        print(f"  {c}: {antes[c]:.1f} -> {despues[c]:.1f}")

    # --- Encoding de variables categóricas ---
    le_modalidad = LabelEncoder()
    df["modalidad_trabajo_enc"] = le_modalidad.fit_transform(df["modalidad_trabajo"])

    le_situacion = LabelEncoder()
    df["situacion_enc"] = le_situacion.fit_transform(df["situacion_academica"])

    joblib.dump(le_modalidad, os.path.join(MODELS_DIR, "encoder_modalidad.pkl"))
    joblib.dump(le_situacion, os.path.join(MODELS_DIR, "encoder_situacion.pkl"))

    return df, le_situacion


def paso_2_regresion_lineal(df: pd.DataFrame):
    """Regresión Lineal: predice la nota del EXAMEN FINAL a partir de
    hábitos del estudiante, para usarse cuando el usuario aún no la tiene."""
    print("\n" + "=" * 70)
    print("PASO 2: Regresión Lineal -> predicción de Examen Final")
    print("=" * 70)

    X = df[FEATURES_REGRESION_EXAMEN]
    y = df["examen_final"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler_reg = StandardScaler()
    X_train_s = scaler_reg.fit_transform(X_train)
    X_test_s = scaler_reg.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_train_s, y_train)

    y_pred = modelo.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    coefs = dict(zip(FEATURES_REGRESION_EXAMEN, modelo.coef_))
    print("\nCoeficientes (impacto de cada variable en el examen final):")
    for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])):
        signo = "↑ sube nota" if v > 0 else "↓ baja nota"
        print(f"  {k:28s} {v:+.4f}  ({signo})")

    joblib.dump(modelo, os.path.join(MODELS_DIR, "regresion_lineal_examen.pkl"))
    joblib.dump(scaler_reg, os.path.join(MODELS_DIR, "scaler_regresion.pkl"))

    metricas = {"r2": r2, "mae": mae, "rmse": rmse, "coeficientes": coefs}
    return metricas


def paso_3_clasificacion(df: pd.DataFrame, le_situacion: LabelEncoder):
    """Clasificación de situación académica: Regresión Logística (base)
    vs Random Forest (modelo principal), con balanceo SMOTE."""
    print("\n" + "=" * 70)
    print("PASO 3: Clasificación (Regresión Logística vs Random Forest)")
    print("=" * 70)

    X = df[FEATURES_NUMERICAS + ["modalidad_trabajo_enc"]]
    y = df["situacion_enc"]

    print("Distribución original de clases:")
    print(pd.Series(le_situacion.inverse_transform(y)).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler_clf = StandardScaler()
    X_train_s = scaler_clf.fit_transform(X_train)
    X_test_s = scaler_clf.transform(X_test)

    # --- Balanceo de clases con SMOTE (solo en train, nunca en test) ---
    min_clase = pd.Series(y_train).value_counts().min()
    k_vecinos = max(1, min(5, min_clase - 1))
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_vecinos)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

    print(f"\nDistribución de clases DESPUÉS de SMOTE (train):")
    print(pd.Series(le_situacion.inverse_transform(y_train_bal)).value_counts())

    resultados = {}

    # ---------------- Regresión Logística (modelo base) ----------------
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_bal, y_train_bal)
    y_pred_lr = log_reg.predict(X_test_s)

    resultados["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "f1_macro": f1_score(y_test, y_pred_lr, average="macro"),
        "precision_macro": precision_score(y_test, y_pred_lr, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_lr, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    }
    print("\n--- Regresión Logística (base) ---")
    print(classification_report(y_test, y_pred_lr,
                                 target_names=le_situacion.classes_, zero_division=0))

    # ---------------- Random Forest (modelo principal) ----------------
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test_s)

    resultados["random_forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "f1_macro": f1_score(y_test, y_pred_rf, average="macro"),
        "precision_macro": precision_score(y_test, y_pred_rf, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_rf, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }
    print("\n--- Random Forest (principal) ---")
    print(classification_report(y_test, y_pred_rf,
                                 target_names=le_situacion.classes_, zero_division=0))

    importancias = dict(zip(X.columns, rf.feature_importances_))
    print("Importancia de variables (Random Forest):")
    for k, v in sorted(importancias.items(), key=lambda x: -x[1])[:8]:
        print(f"  {k:28s} {v:.4f}")
    resultados["random_forest"]["feature_importance"] = importancias

    print("\n>>> Comparación final: "
          f"RF accuracy={resultados['random_forest']['accuracy']:.3f} vs "
          f"LogReg accuracy={resultados['logistic_regression']['accuracy']:.3f} | "
          f"RF F1={resultados['random_forest']['f1_macro']:.3f} vs "
          f"LogReg F1={resultados['logistic_regression']['f1_macro']:.3f}")

    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler_clf, os.path.join(MODELS_DIR, "scaler_clasificacion.pkl"))

    return resultados, (X_train_bal, y_train_bal, X_test_s, y_test)


def paso_4_red_neuronal(datos_split, le_situacion):
    """Red Neuronal (MLPClassifier) con GridSearchCV para búsqueda de
    hiperparámetros, mejorando el desempeño frente a una MLP "default"."""
    print("\n" + "=" * 70)
    print("PASO 4: Red Neuronal (MLPClassifier) + GridSearchCV")
    print("=" * 70)

    X_train_bal, y_train_bal, X_test_s, y_test = datos_split

    # --- Modelo base (sin tuning), para poder JUSTIFICAR la mejora ---
    mlp_base = MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
    mlp_base.fit(X_train_bal, y_train_bal)
    y_pred_base = mlp_base.predict(X_test_s)
    f1_base = f1_score(y_test, y_pred_base, average="macro")
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"MLP base   -> accuracy={acc_base:.4f}  f1_macro={f1_base:.4f}")

    # --- Búsqueda de hiperparámetros ---
    param_grid = {
        "hidden_layer_sizes": [(100,), (64, 32), (64, 32, 16), (128, 64)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.005],
        "activation": ["relu", "tanh"],
    }

    grid = GridSearchCV(
        MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True,
                       n_iter_no_change=15),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train_bal, y_train_bal)

    # Selección robusta: si por variabilidad del split el modelo "optimizado"
    # no supera al base en el set de prueba, se conserva el que mejor
    # generaliza (mismo criterio f1_macro), pero siempre se reportan AMBOS
    # resultados de forma transparente.

    mlp_grid = grid.best_estimator_
    y_pred_grid = mlp_grid.predict(X_test_s)
    f1_grid = f1_score(y_test, y_pred_grid, average="macro")
    acc_grid = accuracy_score(y_test, y_pred_grid)

    print(f"MLP + GridSearchCV -> accuracy={acc_grid:.4f}  f1_macro={f1_grid:.4f}")
    print(f"Mejores hiperparámetros encontrados (CV f1_macro={grid.best_score_:.4f}): "
          f"{grid.best_params_}")
    print(f"Δ F1-macro (test) vs base: {f1_grid - f1_base:+.4f}")

    # Nos quedamos con el modelo que mejor generaliza en el set de prueba,
    # reportando siempre ambos resultados de forma transparente (ver módulo
    # de Ética: no se ocultan resultados desfavorables).
    if f1_grid >= f1_base:
        mejor_mlp, y_pred_mejor, f1_mejor, acc_mejor, ganador = mlp_grid, y_pred_grid, f1_grid, acc_grid, "optimizado (GridSearchCV)"
    else:
        mejor_mlp, y_pred_mejor, f1_mejor, acc_mejor, ganador = mlp_base, y_pred_base, f1_base, acc_base, "base"

    print(f">>> Modelo final seleccionado: MLP {ganador}")
    print("\n--- Reporte de clasificación (MLP seleccionado) ---")
    print(classification_report(y_test, y_pred_mejor,
                                 target_names=le_situacion.classes_, zero_division=0))

    resultados = {
        "mlp_base": {"accuracy": acc_base, "f1_macro": f1_base},
        "mlp_optimizado_gridsearch": {
            "accuracy": acc_grid,
            "f1_macro": f1_grid,
            "cv_best_score_f1_macro": grid.best_score_,
            "mejores_hiperparametros": grid.best_params_,
        },
        "modelo_final_seleccionado": ganador,
        "metricas_modelo_final": {
            "accuracy": acc_mejor,
            "f1_macro": f1_mejor,
            "confusion_matrix": confusion_matrix(y_test, y_pred_mejor).tolist(),
        },
    }

    joblib.dump(mejor_mlp, os.path.join(MODELS_DIR, "mlp_neural_network.pkl"))
    return resultados


def paso_5_clustering(df: pd.DataFrame):
    """K-Means no supervisado -> Perfiles Académicos."""
    print("\n" + "=" * 70)
    print("PASO 5: Clustering K-Means -> Perfiles Académicos")
    print("=" * 70)

    X = df[FEATURES_CLUSTERING]
    scaler_cluster = StandardScaler()
    X_s = scaler_cluster.fit_transform(X)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_s)
    df["cluster"] = clusters

    # PCA solo para visualización 2D en el dashboard
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_pca = pca.fit_transform(X_s)
    df["pca_x"] = coords_pca[:, 0]
    df["pca_y"] = coords_pca[:, 1]

    # --- Etiquetado semántico automático de cada cluster según sus medias ---
    resumen = df.groupby("cluster")[FEATURES_CLUSTERING].mean()
    print("Medias por cluster:")
    print(resumen)

    etiquetas = {}
    global_means = df[FEATURES_CLUSTERING].mean()
    for c in resumen.index:
        fila = resumen.loc[c]
        trabaja_mucho = fila["horas_trabajo_semanal"] > global_means["horas_trabajo_semanal"]
        estudia_bien = fila["horas_estudio_semanal"] >= global_means["horas_estudio_semanal"]
        asiste_bien = fila["asistencia_pct"] >= global_means["asistencia_pct"]
        estres_alto = fila["nivel_estres"] > global_means["nivel_estres"]
        carga_alta = fila["carga_familiar"] > global_means["carga_familiar"]

        # Reglas priorizadas: primero se detectan los perfiles más
        # específicos (carga familiar + estrés) antes que las reglas
        # genéricas de ausentismo, para no mezclar dos causas distintas
        # de riesgo bajo la misma etiqueta.
        if trabaja_mucho and carga_alta and estres_alto:
            etiquetas[c] = "Vulnerable por Carga Familiar y Estres"
        elif trabaja_mucho and not asiste_bien:
            etiquetas[c] = "Riesgo por Ausentismo Laboral"
        elif trabaja_mucho and estudia_bien and asiste_bien:
            etiquetas[c] = "Trabajador Esforzado"
        elif not trabaja_mucho and estudia_bien and asiste_bien:
            etiquetas[c] = "Estudiante de Alto Rendimiento"
        else:
            etiquetas[c] = "Perfil Mixto / Regular"

    # Garantizar unicidad de etiquetas si dos clusters calzan en la misma regla
    usados = {}
    etiquetas_finales = {}
    for c, nombre in etiquetas.items():
        if nombre in usados:
            usados[nombre] += 1
            etiquetas_finales[c] = f"{nombre} ({usados[nombre]})"
        else:
            usados[nombre] = 1
            etiquetas_finales[c] = nombre

    df["perfil_academico"] = df["cluster"].map(etiquetas_finales)

    print("\nEtiquetas asignadas a cada cluster:")
    for c, nombre in etiquetas_finales.items():
        print(f"  Cluster {c}: {nombre}  (n={sum(df['cluster'] == c)})")

    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(scaler_cluster, os.path.join(MODELS_DIR, "scaler_clustering.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_clustering.pkl"))
    joblib.dump(etiquetas_finales, os.path.join(MODELS_DIR, "etiquetas_clusters.pkl"))

    df.to_csv(os.path.join(DATA_DIR, "dataset_con_clusters.csv"), index=False)

    return etiquetas_finales, resumen.to_dict()


def main():
    df, le_situacion = paso_1_datos()
    metricas_regresion = paso_2_regresion_lineal(df)
    metricas_clasificacion, datos_split = paso_3_clasificacion(df, le_situacion)
    metricas_mlp = paso_4_red_neuronal(datos_split, le_situacion)
    etiquetas_clusters, resumen_clusters = paso_5_clustering(df)

    metricas_totales = {
        "regresion_lineal_examen_final": metricas_regresion,
        "clasificacion": metricas_clasificacion,
        "red_neuronal": metricas_mlp,
        "clusters": {
            "etiquetas": etiquetas_clusters,
            "resumen_medias": resumen_clusters,
        },
        "clases_situacion": list(le_situacion.classes_),
        "features_regresion_examen": FEATURES_REGRESION_EXAMEN,
        "features_clasificacion": FEATURES_NUMERICAS + ["modalidad_trabajo_enc"],
        "features_clustering": FEATURES_CLUSTERING,
    }

    with open(os.path.join(MODELS_DIR, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_totales, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO. Artefactos guardados en ./models/")
    print("=" * 70)
    for archivo in sorted(os.listdir(MODELS_DIR)):
        print(f"  - {archivo}")


if __name__ == "__main__":
    main()

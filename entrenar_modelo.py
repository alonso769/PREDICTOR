"""
Entrenamiento del modelo predictivo - Situación Académica UPN
Incluye features de notas T1/T2 integradas.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'estudiantes_upn.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_upn.pkl')
META_PATH  = os.path.join(BASE_DIR, 'models', 'modelo_meta.json')

# ── FEATURES (ahora incluyen notas T1/T2) ────────────────────────────
FEATURES_NUM = [
    'semestre',
    'cursos_matriculados',
    'horas_trabajo_semanal',
    'asistencia_pct',
    'horas_estudio_semanal',
    'promedio_ponderado',
    'cursos_jalados_acumulados',
    'entrega_trabajos_pct',
    'participacion_clase',
    'distancia_campus_km',
    'internet_estable',
    'apoyo_familiar',
    'uso_tutoria',
    'deuda_pension',
    # ── Notas actuales ──
    'promedio_t1',
    'promedio_t2',
    'cursos_en_riesgo_notas',
    'pct_creditos_riesgo',
]
FEATURES_CAT = ['modalidad_trabajo', 'turno_clases']
TARGET       = 'situacion_academica'
LABEL_ORDER  = ['APROBADO_REGULAR', 'EN_RIESGO', 'RIESGO_ALTO', 'DESERCION_PROBABLE']


def cargar_datos():
    if not os.path.exists(DATA_PATH):
        print("⚠️  Dataset no encontrado. Generando uno nuevo...")
        import generar_dataset
        df = generar_dataset.generar_dataset(1200)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    # Compatibilidad: si el CSV viejo no tiene las nuevas columnas, las estimamos
    for col in ['promedio_t1', 'promedio_t2']:
        if col not in df.columns:
            df[col] = df['promedio_ponderado'] + np.random.normal(0, 1.5, len(df))
            df[col] = df[col].clip(0, 20).round(2)
    if 'cursos_en_riesgo_notas' not in df.columns:
        df['cursos_en_riesgo_notas'] = (
            df['cursos_matriculados'] * ((11 - df['promedio_t1']).clip(0) / 11)
        ).round().astype(int).clip(0, df['cursos_matriculados'])
    if 'pct_creditos_riesgo' not in df.columns:
        df['pct_creditos_riesgo'] = (
            df['cursos_en_riesgo_notas'] / df['cursos_matriculados']
        ).round(2)

    print(f"✅ Dataset cargado: {len(df)} registros")
    print(df[TARGET].value_counts())
    return df


def construir_pipeline():
    preprocesador = ColumnTransformer(transformers=[
        ('num', StandardScaler(), FEATURES_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT),
    ])
    modelo_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([('preprocesador', preprocesador), ('modelo', modelo_rf)])


def entrenar():
    df       = cargar_datos()
    X        = df[FEATURES_NUM + FEATURES_CAT]
    y        = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n🔄 Entrenando modelo...")
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 Accuracy: {acc:.4f}  |  F1-weighted: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABEL_ORDER, zero_division=0))

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores= cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV-5 Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        'fecha_entrenamiento': datetime.now().isoformat(),
        'n_registros':         len(df),
        'accuracy_test':       round(acc, 4),
        'f1_weighted':         round(f1, 4),
        'cv_accuracy_mean':    round(float(cv_scores.mean()), 4),
        'cv_accuracy_std':     round(float(cv_scores.std()), 4),
        'features_num':        FEATURES_NUM,
        'features_cat':        FEATURES_CAT,
        'clases':              LABEL_ORDER,
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {MODEL_PATH}")
    return pipeline, meta


if __name__ == '__main__':
    entrenar()

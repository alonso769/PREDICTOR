# Sistema Predictivo de Rendimiento Académico
### Estudiantes universitarios que trabajan — Proyecto final, Sistemas Inteligentes

## Estructura del proyecto
```
sistema_predictivo/
├── generador_datos.py     # Generador de dataset sintético (6000 estudiantes)
├── entrenamiento.py        # Pipeline completo de ML (regresión, clasificación,
│                            #   red neuronal, clustering)
├── app.py                  # Dashboard interactivo (Streamlit)
├── requirements.txt
├── data/                    # Se crea automáticamente al ejecutar
│   ├── dataset_estudiantes.csv
│   └── dataset_con_clusters.csv
└── models/                  # Se crea automáticamente al ejecutar
    ├── regresion_lineal_examen.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── mlp_neural_network.pkl
    ├── kmeans.pkl
    ├── scaler_*.pkl, encoder_*.pkl, pca_clustering.pkl
    └── metricas.json
```

## Cómo ejecutar (orden obligatorio)

```bash
# 1. Crear entorno e instalar dependencias
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generar datos y entrenar TODOS los modelos (una sola vez)
python entrenamiento.py
#    (esto ejecuta internamente generador_datos.py si no existe el dataset)

# 3. Levantar el dashboard
streamlit run app.py
```

Se abrirá en `http://localhost:8501`.

## ¿Qué hace cada modelo?

| Modelo | Tipo | Objetivo | Ubicación |
|---|---|---|---|
| Regresión Lineal | Supervisado (regresión) | Predecir nota de Examen Final si el usuario no la tiene aún | `regresion_lineal_examen.pkl` |
| Regresión Logística | Supervisado (clasificación, modelo base) | Situación Académica: Aprobado / Riesgo / Deserción | `logistic_regression.pkl` |
| Random Forest | Supervisado (clasificación, modelo principal) | Situación Académica (comparado contra Reg. Logística) | `random_forest.pkl` |
| MLPClassifier (+ GridSearchCV) | Deep Learning / Red Neuronal | Situación Académica, con búsqueda de hiperparámetros | `mlp_neural_network.pkl` |
| K-Means | No supervisado | Agrupar estudiantes en "Perfiles Académicos" | `kmeans.pkl` |

## Lógica de negocio validada
- ↑ horas de trabajo y ↓ horas de estudio → ↓ nota final y ↑ probabilidad de Riesgo/Deserción.
- ↑ entendimiento del curso y asistencia → compensan parcialmente el efecto de trabajar
  muchas horas (ver pestaña "Ética en la IA" en el dashboard).
- El dataset incluye un grupo de contraste de estudiantes que NO trabajan, para que los
  modelos aprendan el efecto real del trabajo y no penalicen la condición laboral en sí misma.

## Notas para la sustentación
- Las métricas comparativas (Regresión Logística vs Random Forest, MLP base vs optimizado)
  se generan automáticamente en `models/metricas.json` y se visualizan en la pestaña
  "📊 Comparación de Modelos" del dashboard.
- El balanceo de clases con SMOTE se aplica **solo** sobre el conjunto de entrenamiento,
  nunca sobre el de prueba, para evitar fuga de datos (data leakage).
- Si necesitas regenerar todo desde cero (por ejemplo, para variar la semilla aleatoria),
  borra las carpetas `data/` y `models/` y vuelve a ejecutar `python entrenamiento.py`.

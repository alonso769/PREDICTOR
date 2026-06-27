# 🎓 Predictor de Situación Académica - UPN MGT

Sistema de machine learning para predecir la situación académica de estudiantes
de la **Universidad Privada del Norte** en la modalidad para gente que trabaja.

## Clases que predice el modelo

| Clase | Descripción |
|---|---|
| ✅ APROBADO_REGULAR | El estudiante va bien, sin riesgos visibles |
| ⚠️ EN_RIESGO | Posible jale de algún curso si no mejora |
| 🔴 RIESGO_ALTO | Probable jale de varios cursos, requiere intervención |
| 🚨 DESERCION_PROBABLE | Alta probabilidad de abandono del ciclo |

## Variables que usa el modelo

| Variable | Descripción |
|---|---|
| `semestre` | Semestre actual (1-10) |
| `cursos_matriculados` | Número de cursos este ciclo |
| `horas_trabajo_semanal` | Horas trabajadas por semana |
| `modalidad_trabajo` | presencial / remoto / mixto / sin_trabajo |
| `turno_clases` | noche / sabado_domingo |
| `asistencia_pct` | % de asistencia a clases |
| `horas_estudio_semanal` | Horas de estudio autónomo por semana |
| `promedio_ponderado` | Promedio ponderado actual (0-20) |
| `cursos_jalados_acumulados` | Cursos jalados en toda la carrera |
| `entrega_trabajos_pct` | % de trabajos entregados a tiempo |
| `participacion_clase` | Nivel de participación (1-5) |
| `distancia_campus_km` | Distancia del hogar al campus |
| `internet_estable` | Acceso a internet confiable (0/1) |
| `apoyo_familiar` | Apoyo del entorno familiar (1-5) |
| `uso_tutoria` | Usa el servicio de tutoría UPN (0/1) |
| `deuda_pension` | Meses de deuda en pensión (0-3) |

## Modelo

Ensemble **VotingClassifier (soft)** compuesto por:
- `RandomForestClassifier` (300 árboles, balanceo de clases)
- `GradientBoostingClassifier` (200 estimadores)

Pipeline completo con:
- `StandardScaler` para variables numéricas
- `OneHotEncoder` para variables categóricas
- Validación cruzada estratificada 5-fold

## Instalación y uso

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la aplicación
streamlit run app.py
```

La app abrirá en `http://localhost:8501`

## Flujo de trabajo

1. **⚙️ Entrenar Modelo** → Genera dataset sintético → Entrena el modelo
2. **🔮 Predecir** → Ingresa datos de un estudiante → Obtén predicción + recomendaciones
3. **➕ Agregar Caso** → Agrega estudiantes reales al dataset
4. Repite el entrenamiento con más datos para mejorar precisión

## Estructura del proyecto

```
upn_predictor/
├── app.py                  # Aplicación Streamlit principal
├── generar_dataset.py      # Generador de datos sintéticos
├── entrenar_modelo.py      # Script de entrenamiento
├── requirements.txt
├── data/
│   └── estudiantes_upn.csv # Dataset (se crea al entrenar)
├── models/
│   ├── modelo_upn.pkl      # Modelo entrenado
│   └── modelo_meta.json    # Métricas del último entrenamiento
└── README.md
```

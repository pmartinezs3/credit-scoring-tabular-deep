# Credit Scoring Tabular — DNN vs ResNet + SHAP

Proyecto de **scoring crediticio** usando el dataset *German Credit* (UCI/Kaggle). 
Compara una **Red Densa (DNN)** con una **ResNet tabular** (skip connections) e incluye **SMOTE** para balanceo y **SHAP** para explicabilidad.

## Contenido
- `Evaluacion.py` → script principal (carga, preprocesamiento, modelos, evaluación, costo de negocio y SHAP).
- `reports/` → coloca aquí las figuras generadas. Incluí ejemplos de tu corrida.
- `data/` → (opcional) si quieres usar un CSV local en lugar de descargar desde UCI.
- `requirements.txt`, `.gitignore`, `LICENSE` y este `README.md`.

## Requisitos
- Python 3.10+
- CPU funciona. (GPU opcional para acelerar TensorFlow)

```bash
# Crear entorno (Windows)
python -m venv .venv
.\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar
Por defecto, `Evaluacion.py` intentará descargar el dataset de UCI si no encuentra un CSV local.
Si tienes un CSV propio, edita la variable `CSV_PATH` al inicio del archivo para señalar la ruta.

```bash
# Opción 1: ejecutar así tal cual (descarga UCI si es posible)
python Evaluacion.py

# Opción 2: editar Evaluacion.py y poner CSV_PATH = "data/german.csv" (o el nombre que tengas) y luego:
python Evaluacion.py
```

### Salidas esperadas
- Métricas en consola (Accuracy, Precision, Recall, F1, ROC-AUC) para **DNN** y **ResNetTab**.
- Figuras: **curva ROC**, **matriz de confusión**, **distribución de clases** y **gráficos SHAP**.
- Cálculo de **costo de negocio** con pesos para FP/FN.

Las imágenes de ejemplo en `reports/` muestran una corrida típica con AUC ~0.78 para DNN.

## Notas
- El script usa `SMOTE` por defecto para balancear el entrenamiento. Si prefieres `class_weight`, cambia `USE_SMOTE = False`.
- Para acelerar SHAP, puedes reducir `nsamples` o usar `shap.kmeans` como background resumido.

## Licencia
MIT © Pamela Martinez

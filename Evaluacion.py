#!/usr/bin/env python
# coding: utf-8

"""
Módulo 7 — Evaluación
Sistema inteligente de scoring crediticio con redes profundas (German Credit)

Cumple con:
1) Carga y análisis exploratorio
2) Preprocesamiento (normalización, codificación, desbalanceo via pesos o SMOTE)
3) Modelado: DNN base vs. ResNet tabular (skip connections)
4) Explicabilidad: SHAP (y opcional LIME)
5) Evaluación: accuracy, precision, recall, F1, ROC-AUC, matriz de confusión
6) Reflexión crítica: costos de error, sesgos, explicabilidad
"""

# =====================
# Importaciones y setup
# =====================
import os
os.environ.pop("KERAS_BACKEND", None)  

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

import shap  

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================
# Parámetros
# =====================
CSV_PATH = None 
TARGET_POSITIVE = "bad"  
USE_SMOTE = True          
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Modelo
L2 = 1e-4
DROPOUT = 0.3
EPOCHS = 30
BATCH_SIZE = 256
LR1 = 1e-3  
LR2 = 5e-4 

# =====================
# Carga de datos
# =====================

def load_german_credit(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Búsqueda de nombres típicos
    for name in ["german_credit_data.csv", "GermanCreditData.csv", "German_Credit_Data.csv"]:
        if os.path.exists(name):
            return pd.read_csv(name)

    # Descarga UCI (Statlog) si no hay CSV local
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        cols = [f"col_{i}" for i in range(1, 21)] + ["target"]
        df = pd.read_csv(url, sep=' ', header=None, names=cols)
        return df
    except Exception:
        raise FileNotFoundError("No se encontró CSV local ni fue posible acceder a UCI. Coloca el archivo junto al script y reintenta.")

print("Cargando dataset…")
df = load_german_credit(CSV_PATH)
print("Dimensiones:", df.shape)
print(df.head(3))

# =====================
# Limpieza + formateo de target
# =====================

def prepare_target(df: pd.DataFrame):
    df = df.copy()
    y = None
    if "Risk" in df.columns:  # Kaggle version: 'good'/'bad'
        y = (df["Risk"].str.lower() == TARGET_POSITIVE.lower()).astype(int).values
        X = df.drop(columns=["Risk"])  # resto de columnas como features
    elif "target" in df.columns:  # UCI Statlog: 1=good, 2=bad
        y = (df["target"].astype(int) == 2).astype(int).values
        X = df.drop(columns=["target"])
    else:
        # fallback: si no hay nombre, asumimos última columna es target
        y_last = df.iloc[:, -1].astype(str)
        if set(y_last.unique()) <= set(["good","bad","Good","Bad"]):
            y = (y_last.str.lower() == TARGET_POSITIVE.lower()).astype(int).values
        else:
            y = (df.iloc[:, -1].astype(int) == 2).astype(int).values
        X = df.iloc[:, :-1]
    return X, y

X_raw, y = prepare_target(df)
print("Positivos (impago=1):", int(y.sum()), "/", y.size)

# =====================
# Análisis exploratorio básico
# =====================

print("Tipos de datos:\n", X_raw.dtypes.value_counts())

# Distribución de clases
plt.figure()
vals, counts = np.unique(y, return_counts=True)
plt.bar([str(v) for v in vals], counts)
plt.title("Distribución de clases (0=No impago, 1=Impago)")
plt.xlabel("Clase"); plt.ylabel("Frecuencia")
plt.tight_layout()

# =====================
# Preprocesamiento
# =====================

cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == 'object']
num_cols = [c for c in X_raw.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

# Split train/val/test con estratificación
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=42, stratify=y
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=VAL_SIZE, random_state=42, stratify=y_train
)

# Ajuste y transformación
X_train = preprocess.fit_transform(X_train_raw)
X_val   = preprocess.transform(X_val_raw)
X_test  = preprocess.transform(X_test_raw)

# Nombres de features
feature_names_num = num_cols
ohe_names = []
if cat_cols:
    ohe = preprocess.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
feature_names = feature_names_num + ohe_names

# Balanceo
class_weights = None
from imblearn.over_sampling import RandomOverSampler
if USE_SMOTE:
    try:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("SMOTE aplicado:", X_train.shape)
    except Exception as e:
        print("SMOTE falló, usando RandomOverSampler. Motivo:", repr(e))
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print("ROS aplicado:", X_train.shape)
else:
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(k): float(v) for k, v in zip(classes, cw)}
    print("Pesos de clase:", class_weights)

# =====================
# Modelos
# =====================

def make_dnn(input_dim):
    reg = regularizers.l2(L2)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dropout(DROPOUT),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(DROPOUT),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LR1),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])
    return model

def residual_block(x, units, reg):
    skip = x
    out = layers.Dense(units, activation=None, kernel_regularizer=reg)(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Dropout(DROPOUT)(out)
    out = layers.Dense(units, activation=None, kernel_regularizer=reg)(out)
    out = layers.BatchNormalization()(out)
    if skip.shape[-1] != units:
        skip = layers.Dense(units, activation=None, kernel_regularizer=reg)(skip)
    out = layers.Add()([out, skip])
    out = layers.ReLU()(out)
    return out

def make_resnet_tabular(input_dim):
    reg = regularizers.l2(L2)
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(inp)
    x = residual_block(x, 128, reg)
    x = residual_block(x, 64, reg)
    x = residual_block(x, 64, reg)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out, name='resnet_tabular')
    model.compile(optimizer=tf.keras.optimizers.Adam(LR1),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])
    return model

cbs = [
    callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', min_lr=1e-5)
]

print("\\nEntrenando DNN…")
dnn = make_dnn(X_train.shape[1])
dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=2,
    class_weight=class_weights
)

print("\\nEntrenando ResNet tabular…")
resnet = make_resnet_tabular(X_train.shape[1])
resnet.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=2,
    class_weight=class_weights
)

# =====================
# Evaluación y curvas
# =====================

def evaluate_model(model, name):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\\n{name} – Test metrics:\\n  Acc: {acc:.4f}\\n  Precision: {prec:.4f}\\n  Recall: {rec:.4f}\\n  F1: {f1:.4f}\\n  ROC-AUC: {auc:.4f}")
    print("\\nClassification report:\\n", classification_report(y_test, y_pred, digits=4))

    # ROC
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {name}")
    plt.legend(); plt.tight_layout(); plt.savefig(f"reports/roc_{name}.png", dpi=180)

    # Matriz de confusión
    plt.figure();
    plt.imshow(cm, cmap='viridis'); plt.colorbar();
    plt.title(f"Matriz de confusión – {name}");
    plt.xlabel("Predicción"); plt.ylabel("Real");
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout(); plt.savefig(f"reports/cm_{name}.png", dpi=180)

    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "auc":auc, "cm":cm, "y_prob":y_prob, "y_pred":y_pred}

res_dnn = evaluate_model(dnn, "DNN")
res_res = evaluate_model(resnet, "ResNetTab")

# =====================
# Costeo de errores
# =====================
COST_FP = 1.0
COST_FN = 5.0

def business_cost(cm, cost_fp=COST_FP, cost_fn=COST_FN):
    tn, fp, fn, tp = cm.ravel()
    return fp*cost_fp + fn*cost_fn

cost_dnn = business_cost(res_dnn["cm"]) 
cost_res = business_cost(res_res["cm"])
print(f"\\nCosto de negocio (relativo) – DNN: {cost_dnn:.1f} | ResNetTab: {cost_res:.1f}")

# =====================
# Explicabilidad con SHAP (muestra)
# =====================
print("\\nCalculando explicabilidad SHAP sobre una muestra…")
idx_bg = np.random.choice(X_train.shape[0], size=min(200, X_train.shape[0]), replace=False)
X_bg = X_train[idx_bg]
predict_fn_dnn = lambda data: dnn.predict(data, verbose=0).ravel()
predict_fn_res = lambda data: resnet.predict(data, verbose=0).ravel()

explainer_dnn = shap.KernelExplainer(predict_fn_dnn, X_bg)
explainer_res = shap.KernelExplainer(predict_fn_res, X_bg)

X_sample = X_test[:100]
shap_values_dnn = explainer_dnn.shap_values(X_sample, nsamples=100)
shap_values_res = explainer_res.shap_values(X_sample, nsamples=100)

# summary plots
try:
    import os
    os.makedirs("reports", exist_ok=True)
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values_dnn, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP summary – DNN"); plt.tight_layout(); plt.savefig("reports/shap_summary_dnn.png", dpi=180); plt.close()
    shap.summary_plot(shap_values_res, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP summary – ResNetTab"); plt.tight_layout(); plt.savefig("reports/shap_summary_resnet.png", dpi=180); plt.close()
except Exception as e:
    print("Aviso: no fue posible guardar gráficos SHAP:", repr(e))

print("\\nListo. Figuras guardadas en carpeta reports/.")

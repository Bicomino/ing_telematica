import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DE LA ARQUITECTURA ANN
# ==========================================
input_dim = X_train.shape[1]

model = Sequential([
    Dense(units=64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 2. CONFIGURACIÓN DE EARLY STOPPING
# ==========================================
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# ==========================================
# 3. ENTRENAMIENTO (Simulado)
# ==========================================
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])

# ==========================================
# 4. PREDICCIÓN Y ANÁLISIS DE DIMENSIONALIDAD
# ==========================================
# Obtenemos las predicciones para el set de prueba
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype(int).flatten()

# --- Escalar datos (requerido para PCA y t-SNE) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test)

# --- PCA: Análisis de Varianza ---
pca_full = PCA()
pca_full.fit(X_scaled)
varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

# Determinar componentes para el 90% de varianza
n_comp_90 = np.argmax(varianza_acumulada >= 0.9) + 1
print(f"Número de componentes para 90% energía: {n_comp_90}")

# --- Visualizar PCA a 2 componentes ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PCA1': X_pca2[:, 0],
    'PCA2': X_pca2[:, 1],
    'Clase': y_pred
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Clase', palette='viridis', alpha=0.6)
plt.title('Clasificación en Espacio PCA (2D) - Resultados ANN')
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# --- t-SNE: Análisis de proximidad ---
pca_90 = PCA(n_components=n_comp_90)
X_pca_90 = pca_90.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_pca_90)

tsne_df = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'Clase': y_pred
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Clase', palette='magma', alpha=0.6)
plt.title('Visualización con t-SNE (Tras PCA 90%) - Resultados ANN')
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
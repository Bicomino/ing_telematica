import tensorflow as tf
from tensorflow import keras
import pathlib

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS (RGB) ---
# ResNet50 requiere imágenes en color (3 canales). 
# Asegúrate de que 'data_dir' contenga carpetas 'legit' y 'phishing'.
data_dir = pathlib.Path("data/processed/favicons_color") 
img_height, img_width = 64, 64
batch_size = 32

# Dataset de entrenamiento (80%)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, 
    validation_split=0.2, 
    subset="training", 
    seed=123,
    image_size=(img_height, img_width), 
    batch_size=batch_size,
    label_mode='binary' # Etiquetas 0/1 para clasificación binaria
)

# Dataset de validación (20%)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=123,
    image_size=(img_height, img_width), 
    batch_size=batch_size,
    label_mode='binary'
)

# --- 2. CONFIGURACIÓN DEL MODELO BASE (TRANSFER LEARNING) ---
# Cargamos ResNet50 sin la cabeza (top=False) para usarlo como extractor de características
pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(64, 64, 3),
    pooling='avg',
    weights='imagenet'
)

# Congelamos los pesos del modelo base para aprovechar el conocimiento previo de ImageNet
for layer in pretrained_model.layers:
    layer.trainable = False

# --- 3. CONSTRUCCIÓN DEL MODELO FINAL ---
resnet_model = keras.Sequential([
    pretrained_model,
    # Capa densa personalizada para adaptar el conocimiento a nuestro problema
    keras.layers.Dense(512, activation='relu'),
    # Salida binaria con Sigmoid (Probabilidad de Phishing)
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilación con optimizador Adam (ideal para Transfer Learning)
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. ENTRENAMIENTO ---
# history = resnet_model.fit(
#     train_ds, 
#     validation_data=val_ds, 
#     epochs=10
# )
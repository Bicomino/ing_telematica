import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# === 1. DEFINICIÓN DEL GENERADOR CONDICIONAL ===
def build_cgenerator(latent_dim, num_classes):
    """
    Construye el Generador Condicional.
    Combina el vector de ruido (latente) con etiquetas One-Hot.
    """
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_classes,)) # Recibe etiquetas One-Hot
    
    # Condicionamiento: Concatenación del ruido y la información de clase
    merged = layers.Concatenate()([noise_input, label_input])

    # Arquitectura base (ejemplo con capas densas, adaptable a Conv2DTranspose)
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    # Suponiendo salida plana para ser reestructurada a imagen después
    out = layers.Dense(64 * 64 * 3, activation="tanh")(x) 
    
    return Model([noise_input, label_input], out, name="cgenerator")

# === 2. LÓGICA DE CARGA O ENTRENAMIENTO PREVIO ===
MODEL_PATH = "models/generator_pretrained.keras"

if os.path.exists(MODEL_PATH):
    print("Cargando generador preentrenado...")
    generator = keras.models.load_model(MODEL_PATH)
else:
    print("Inicializando nuevo generador...")
    generator = build_cgenerator(latent_dim=100, num_classes=2)
    # Lógica de compilación y pre-entrenamiento si fuera necesario
    # generator.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")

# === 3. DISCRIMINADOR CONDICIONAL DESDE CERO ===
def build_cdiscriminator(img_shape, num_classes):
    """
    Construye el Discriminador Condicional.
    Evalúa la autenticidad de la imagen condicionada a su etiqueta.
    """
    img_input = layers.Input(shape=img_shape)
    label_input = layers.Input(shape=(num_classes,))

    # Condicionamiento: Expansión de la etiqueta para que coincida con las dimensiones de la imagen
    # np.prod(img_shape) asegura que el mapa de etiquetas tenga el mismo tamaño que los píxeles
    label_map = layers.Dense(np.prod(img_shape), activation="relu")(label_input)
    label_map = layers.Reshape(img_shape)(label_map)
    
    # Concatenación de la imagen de entrada con el mapa de etiquetas
    merged = layers.Concatenate()([img_input, label_map])

    # Arquitectura de discriminación (ejemplo con capas densas)
    x = layers.Dense(512, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model([img_input, label_input], out, name="cdiscriminator")

# === 4. BUCLE DE ENTRENAMIENTO (LABEL SMOOTHING) ===
def train_step_cgan(generator, discriminator, cgan, X_real, real_labels, half_batch, latent_dim):
    # Generación de etiquetas con suavizado (Label Smoothing)
    # Suavizar etiquetas reales (0.9 - 1.0 en lugar de 1.0)
    y_real = np.ones((half_batch, 1)) - np.random.uniform(0, 0.1, (half_batch, 1))
    
    # Suavizar etiquetas falsas (0.0 - 0.1 en lugar de 0.0)
    y_fake = np.zeros((half_batch, 1)) + np.random.uniform(0, 0.1, (half_batch, 1))

    # Entrenamiento del Discriminador (D)
    # d_loss_real = discriminator.train_on_batch([X_real, real_labels], y_real)
    
    # Entrenamiento del Generador (G) a través del modelo combinado
    # g_loss = cgan.train_on_batch([noise, sampled_labels], y_gan)
    pass
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# === 1. ARQUITECTURA DEL GENERADOR (cGAN) ===
def build_generator(latent_dim, num_classes):
    """
    Generador condicional: Recibe ruido aleatorio y una etiqueta de clase.
    Utiliza la etiqueta para filtrar el ruido y generar muestras específicas.
    """
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,), dtype="int32")

    # Condicionamiento: Embedding de la etiqueta al tamaño del espacio latente
    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Combinación de ruido y clase mediante multiplicación
    joined_input = layers.Multiply()([noise_input, label_embedding])

    # Estructura de la red densa
    x = layers.Dense(128, activation="relu")(joined_input)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(64, activation="tanh")(x) # Ajustar según dimensión de salida

    return Model([noise_input, label_input], out, name="generator")

# === 2. ARQUITECTURA DEL DISCRIMINADOR (cGAN) ===
def build_discriminator(img_shape, num_classes):
    """
    Discriminador condicional: Evalúa la autenticidad del par (dato, etiqueta).
    """
    img_input = Input(shape=img_shape)
    label_input = Input(shape=(1,), dtype="int32")

    # Condicionamiento: Expandir etiqueta para que coincida con la forma del dato
    label_embedding = layers.Embedding(num_classes, np.prod(img_shape))(label_input)
    label_embedding = layers.Reshape(img_shape)(label_embedding)

    # Concatenación de la imagen/dato con su etiqueta correspondiente
    merged = layers.Concatenate(axis=-1)([img_input, label_embedding])

    # Capas de clasificación
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model([img_input, label_input], out, name="discriminator")

# === 3. BUCLE DE ENTRENAMIENTO ===
def train_step(X_real, labels_real, half_batch, latent_dim, num_classes):
    """
    Entrenamiento estándar con etiquetas binarias reales (1.0) y falsas (0.0).
    """
    # Etiquetas reales puras (1.0)
    y_real = np.ones((half_batch, 1), dtype=np.float32)
    
    # Entrenamiento del Discriminador con datos reales
    # d_loss_real = discriminator.train_on_batch([X_real, labels_real], y_real)
    
    # Etiquetas para el Generador (objetivo: engañar a D con 1.0)
    y_gan = np.ones((half_batch * 2, 1), dtype=np.float32)
    # g_loss = gan.train_on_batch([noise_g, sampled_labels], y_gan)
    pass
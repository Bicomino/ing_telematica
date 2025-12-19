import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# === 1. GENERADOR NO CONDICIONAL (ncGAN) ===
def build_generator(latent_dim):
    """
    Generador: Crea imágenes sintéticas a partir de ruido aleatorio puro.
    """
    noise_input = layers.Input(shape=(latent_dim,))

    # Arquitectura base
    x = layers.Dense(256, activation="relu")(noise_input)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    
    # Salida: 64x64x3 (RGB) con activación tanh (rango [-1, 1])
    x = layers.Dense(64 * 64 * 3, activation="tanh")(x)
    out = layers.Reshape((64, 64, 3))(x)
    
    return Model(noise_input, out, name="generator_nc")

# === 2. DISCRIMINADOR NO CONDICIONAL (ncGAN) ===
def build_discriminator(img_shape):
    """
    Discriminador: Clasifica imágenes como Reales (1) o Falsas (0).
    """
    img_input = layers.Input(shape=img_shape)

    # Bloques convolucionales para extracción de características visuales
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation="sigmoid")(x) 

    return Model(img_input, out, name="discriminator_nc")

# === 3. ENSAMBLAJE DE LA GAN (Modelo Combinado) ===
latent_dim = 100
img_shape = (64, 64, 3)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Configuración del Discriminador
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
discriminator.trainable = False

# Red GAN combinada (Entrena al Generador a través del Discriminador)
z_input = layers.Input(shape=(latent_dim,))
img = generator(z_input)
validity = discriminator(img)

gan = Model(z_input, validity)
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

# === 4. BUCLE DE ENTRENAMIENTO ESTÁNDAR ===
def train_step_nc(X_real, batch_size, latent_dim):
    half_batch = batch_size // 2
    
    # --- Entrenar Discriminador ---
    # Etiquetas reales puras (1.0)
    y_real = np.ones((half_batch, 1))
    # d_loss_real = discriminator.train_on_batch(X_real, y_real)
    
    # Etiquetas falsas puras (0.0)
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    X_fake = generator.predict(noise)
    y_fake = np.zeros((half_batch, 1))
    # d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)

    # --- Entrenar Generador ---
    noise_full = np.random.normal(0, 1, (batch_size, latent_dim))
    y_gan = np.ones((batch_size, 1)) # Objetivo: que D clasifique lo falso como real
    # g_loss = gan.train_on_batch(noise_full, y_gan)
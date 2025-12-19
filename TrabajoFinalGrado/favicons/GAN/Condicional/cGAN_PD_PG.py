import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# === 1. AUTOENCODER CONDICIONAL (Estrategia de Preentrenamiento) ===
def build_conditional_autoencoder(latent_dim, img_shape, num_classes):
    """
    Crea un Autoencoder que aprende a comprimir y reconstruir imágenes
    condicionadas a su etiqueta de clase.
    """
    # Entrada de imagen y etiqueta
    inp = layers.Input(shape=img_shape)
    label_in = layers.Input(shape=(num_classes,))

    # Condicionamiento espacial: Expandir etiqueta al tamaño de la imagen
    lbl = layers.Dense(np.prod(img_shape), activation="relu")(label_in)
    lbl = layers.Reshape(img_shape)(lbl)

    # Concatenar imagen y etiqueta como un canal adicional
    merged = layers.Concatenate(axis=-1)([inp, lbl])

    # --- ENCODER ---
    x = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(merged)
    x = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim)(x) # Espacio latente condicional

    # --- DECODER (Base del futuro Generador) ---
    d = layers.Dense(8 * 8 * 256, activation='relu')(z)
    d = layers.Reshape((8, 8, 256))(d)
    d = layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu')(d)
    d = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(d)
    out = layers.Conv2D(3, 3, padding='same', activation='tanh')(d) # Salida en rango [-1, 1]

    return Model([inp, label_in], out, name="conditional_autoencoder")

# --- Entrenamiento del Autoencoder ---
# Se utiliza el error cuadrático medio (MSE) para la reconstrucción píxel a píxel
autoencoder = build_conditional_autoencoder(latent_dim=100, img_shape=(64, 64, 3), num_classes=2)
autoencoder.compile(optimizer="adam", loss="mse")

# Entrenamiento (X_all normalizado en [-1, 1])
# autoencoder.fit([X_all, labels_onehot], X_all, epochs=50, batch_size=32)

# === 2. EXTRACCIÓN DEL DECODER COMO GENERADOR CONDICIONAL ===
def extract_generator_from_autoencoder(autoencoder, latent_dim, num_classes):
    """
    Aprovecha las capas entrenadas del decoder para crear el Generador de la GAN.
    """
    latent_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_classes,))

    # Preparar el condicionamiento para el decoder
    lbl = layers.Dense(8 * 8 * 256, activation='relu')(label_input)
    lbl = layers.Reshape((8, 8, 256))(lbl)

    # Extraer las capas específicas del decoder (las últimas 5 capas del autoencoder)
    # Conv2DTranspose y Conv2D final
    decoder_layers = autoencoder.layers[-5:] 

    x = layers.Dense(8 * 8 * 256, activation='relu')(latent_input)
    x = layers.Reshape((8, 8, 256))(x)

    # Fusionar con la etiqueta
    x = layers.Concatenate(axis=-1)([x, lbl])

    # Pasar por las capas preentrenadas
    for layer in decoder_layers:
        x = layer(x)

    return Model([latent_input, label_input], x, name="generator_pretrained")

generator = extract_generator_from_autoencoder(autoencoder, 100, 2)

# === 3. ENSAMBLAJE DE LA cGAN (Discriminador + Generador Preentrenados) ===
# Cargamos el discriminador condicional (asumiendo entrenamiento previo)
# discriminator = keras.models.load_model("models/cdiscriminator_pretrained.h5")
# discriminator.trainable = False

# Definición del modelo combinado GAN
z_input = layers.Input(shape=(100,))
lbl_input = layers.Input(shape=(2,))

fake_img = generator([z_input, lbl_input])
validity = discriminator([fake_img, lbl_input])

cgan = Model([z_input, lbl_input], validity)
cgan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)
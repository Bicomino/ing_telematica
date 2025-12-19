import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# --- 1. DISCRIMINADOR BASADO EN RESNET50 (Pre-entrenamiento) ---
def build_and_train_discriminator(train_ds, val_ds, img_shape, epochs, lr):
    """
    Crea un clasificador binario utilizando ResNet50 como extractor de características.
    Este modelo servirá como base para el discriminador de la GAN.
    """
    base = ResNet50(include_top=False, input_shape=img_shape, pooling='avg', weights='imagenet')
    
    # Congelamos las capas base para mantener el conocimiento de ImageNet
    for layer in base.layers:
        layer.trainable = False 

    # Cabeza densa para clasificación binaria (Legítimo vs Phishing)
    x = layers.Dense(512, activation='relu')(base.output)
    out = layers.Dense(1, activation='sigmoid')(x)

    disc_model = Model(inputs=base.input, outputs=out)
    disc_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Nota: El proceso de fit() se asume ejecutado previamente o en este punto.
    return disc_model

# --- 2. WRAPPER PARA DISCRIMINADOR CONDICIONAL ---
def make_conditional_discriminator(disc_model, img_height, img_width, num_classes):
    """
    Transforma el clasificador ResNet en un Discriminador Condicional.
    Concatena la información de la clase (etiqueta) directamente en el espacio de la imagen.
    """
    # Entrada visual: La GAN suele trabajar en rango [-1, 1]
    img_in = layers.Input(shape=(img_height, img_width, 3))
    # Entrada de etiqueta: One-hot encoding
    lbl_in = layers.Input(shape=(num_classes,))

    # Expandimos la etiqueta para que tenga las mismas dimensiones espaciales que la imagen
    y = layers.Dense(img_height * img_width * 3, activation="relu")(lbl_in)
    y = layers.Reshape((img_height, img_width, 3))(y)

    # Fusión de información: Imagen + Etiqueta
    merged = layers.Concatenate()([img_in, y])

    # Preprocesamiento específico para ResNet:
    # 1. Re-escalar de [-1, 1] a [0, 255]
    # 2. Aplicar la función de normalización de ResNet (Zero-center, etc.)
    x = layers.Lambda(lambda z: (z + 1.0) * 127.5)(merged)
    x = layers.Lambda(lambda z: tf.keras.applications.resnet.preprocess_input(z))(x)

    # Evaluación de validez mediante el modelo ResNet pre-entrenado
    validity = disc_model(x)

    return Model([img_in, lbl_in], validity, name="conditional_discriminator")

# --- 3. CONSTRUCCIÓN DE LA GAN CONDICIONAL (cGAN) ---
# Definición de hiperparámetros y componentes
img_height, img_width = 64, 64
num_classes = 2
latent_dim = 100

# Suponiendo que 'disc_model' y 'generator' ya están definidos/cargados
disc_cond = make_conditional_discriminator(disc_model, img_height, img_width, num_classes)
disc_cond.trainable = False  # Congelar discriminador al entrenar la red combinada (GAN)

# Entradas para el flujo de generación
z_in = layers.Input(shape=(latent_dim,))
lbl_in = layers.Input(shape=(num_classes,))

# 1. El generador crea una imagen a partir del ruido y la etiqueta deseada
img_sintetica = generator([z_in, lbl_in])

# 2. El discriminador condicional evalúa si esa imagen corresponde a esa clase
validity = disc_cond([img_sintetica, lbl_in])

# Modelo GAN final para entrenar el generador
cgan = Model([z_in, lbl_in], validity, name="cgan")
cgan.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
)
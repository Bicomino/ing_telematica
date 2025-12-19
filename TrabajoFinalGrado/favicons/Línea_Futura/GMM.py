import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================
DATA_DIR = "data/processed/favicons_color"
OUT_DIR = "output/gan_gmm_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'models'), exist_ok=True)

IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS_AUTOENCODER = 30
EPOCHS_GAN = 100
SAVE_EVERY = 5

# ============================================================
# 2. SELECCIÓN DE MODELO GMM ÓPTIMO
# ============================================================
def best_gmm(X):
    """
    Ajusta varios modelos GMM y elige el mejor basándose en el criterio BIC.
    """
    X = X.astype(np.float64)
    bic_min = float('inf')
    best_model = None

    for K in range(1, 6):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="diag", 
            reg_covar=1e-3,
            n_init=5
        ).fit(X)

        bic = gmm.bic(X)
        if bic < bic_min:
            bic_min = bic
            best_model = gmm

    print(f"GMM optimizado → Componentes: {best_model.n_components}, BIC: {bic_min:.2f}")
    return best_model

# ============================================================
# 3. CARGA Y NORMALIZACIÓN DEL DATASET
# ============================================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Normalización al rango [-1, 1] para activación tanh
def to_tanh(x, y):
    return (tf.cast(x, tf.float32) / 127.5) - 1., y

train_ds = train_ds.map(to_tanh).unbatch()
X_all = np.array([x.numpy() for x, _ in train_ds])

# ============================================================
# 4. PREENTRENAMIENTO: AUTOENCODER
# ============================================================
def build_autoencoder(latent_dim, img_shape):
    # Encoder
    inp = layers.Input(shape=img_shape)
    x = layers.Conv2D(64, 4, 2, 'same', activation='relu')(inp)
    x = layers.Conv2D(128, 4, 2, 'same', activation='relu')(x)
    x = layers.Conv2D(256, 4, 2, 'same', activation='relu')(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)

    # Decoder
    d = layers.Dense(8 * 8 * 256, activation='relu')(z)
    d = layers.Reshape((8, 8, 256))(d)
    for f in [128, 64, 32]:
        d = layers.Conv2DTranspose(f, 4, 2, 'same', activation='relu')(d)
    out = layers.Conv2D(3, 3, padding='same', activation='tanh')(d)

    return Model(inp, out, name="autoencoder")

autoenc = build_autoencoder(LATENT_DIM, (IMG_SIZE, IMG_SIZE, 3))
autoenc.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae')
autoenc.fit(X_all, X_all, epochs=EPOCHS_AUTOENCODER, batch_size=BATCH_SIZE)

# Extraer Encoder y calcular espacio latente (Z)
encoder = Model(autoenc.input, autoenc.get_layer("latent").output)
Z_all = encoder.predict(X_all, batch_size=64)

# Ajustar GMM al espacio latente obtenido
gmm_model = best_gmm(Z_all)

# Extraer Decoder como Generador
latent_input = keras.Input(shape=(LATENT_DIM,))
x = latent_input
for layer in autoenc.layers[-6:]:
    x = layer(x)
generator = Model(latent_input, x, name="generator")

# ============================================================
# 5. ENSAMBLAJE Y ENTRENAMIENTO DE LA GAN
# ============================================================
# Carga del discriminador (previamente entrenado)
# discriminator = keras.models.load_model("models/discriminator_resnet.h5")
discriminator.trainable = False

# Definición del modelo GAN combinado
z_in = keras.Input(shape=(LATENT_DIM,))
fake_img = generator(z_in)
validity = discriminator(fake_img)
gan = Model(z_in, validity)
gan.compile(optimizer=keras.optimizers.Adam(2e-4, 0.5), loss="binary_crossentropy")

def save_samples(epoch, generator, gmm):
    noise = gmm.sample(16)[0]
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = (gen_imgs + 1) / 2
    plt.figure(figsize=(6,6))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(gen_imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, 'images', f"epoch_{epoch:03d}.png"))
    plt.close()

# Bucle de entrenamiento
for epoch in range(1, EPOCHS_GAN + 1):
    # Entrenar Discriminador
    idx = np.random.randint(0, X_all.shape[0], BATCH_SIZE // 2)
    real_imgs = X_all[idx]
    y_real = np.ones((BATCH_SIZE // 2, 1))
    
    noise = gmm_model.sample(BATCH_SIZE // 2)[0]
    fake_imgs = generator.predict(noise, verbose=0)
    y_fake = np.zeros((BATCH_SIZE // 2, 1))

    d_loss_real = discriminator.train_on_batch(real_imgs, y_real)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, y_fake)

    # Entrenar Generador
    noise = gmm_model.sample(BATCH_SIZE)[0]
    g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

    if epoch % SAVE_EVERY == 0:
        save_samples(epoch, generator, gmm_model)
        print(f"Epoch {epoch}/{EPOCHS_GAN} completada.")
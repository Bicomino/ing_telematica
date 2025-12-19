import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# --- 1. CONFIGURACIÓN Y PREPROCESAMIENTO ---
# Ruta base que contiene las carpetas 'legit' y 'phishing'
base_dir = "data/processed/favicons" 

# Generador con normalización y división para validación (80% train, 20% validation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode='binary', 
    subset='training', 
    shuffle=True, 
    batch_size=32
)

test_gen = datagen.flow_from_directory(
    base_dir, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode='binary', 
    subset='validation', 
    shuffle=False, 
    batch_size=32
)

# --- 2. GESTIÓN DE DATOS Y BALANCEO ---
# Convertir generadores a arrays para entrenamiento directo
X_train = np.concatenate([train_gen[i][0] for i in range(len(train_gen))])
t_train = np.concatenate([train_gen[i][1] for i in range(len(train_gen))])

# Cálculo de pesos de clase para compensar el desbalance de datos
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(t_train),
    y=t_train
)
class_weights = dict(enumerate(weights))

# --- 3. ARQUITECTURA DE LA CNN ---
input_shape = (64, 64, 1) # Imagen de 64x64 en escala de grises

model = keras.models.Sequential([
    # Bloque 1: Convolución y Pooling
    keras.layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Bloque 2: Convolución y Pooling
    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Clasificador denso
    keras.layers.Flatten(),
    keras.layers.Dropout(0.1), # Regularización para evitar overfitting
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid") # Salida binaria (probabilidad)
])

# --- 4. COMPILACIÓN Y ENTRENAMIENTO ---
model.compile(
    loss="binary_crossentropy", 
    optimizer="sgd", 
    metrics=["accuracy"]
)

# Ejemplo de ejecución:
# history = model.fit(
#     X_train, 
#     t_train, 
#     epochs=50, 
#     validation_data=test_gen, 
#     class_weight=class_weights
# )
# Herramienta para la verificación automática de URLs mediante ML en el contexto de ciberseguridad

Este repositorio contiene el ecosistema de desarrollo para un sistema avanzado de detección de phishing mediante **Inteligencia Artificial**.  
El proyecto aborda el problema desde una perspectiva multidisciplinar: análisis de metadatos de URL, características heurísticas del HTML y visión por computador aplicada a activos visuales (**favicons**).
El proyecto obtuvo una calificación de 10 (máximo) con opción a matrícula de honor.

> **⚠️ Aviso importante:**  
> Los códigos presentados en este repositorio son versiones generalizadas y optimizadas de los algoritmos originales. Se han simplificado para mejorar la claridad técnica y asegurar su reproducibilidad por parte de la comunidad. En la [Documentación del proyecto](./docs) podrás encontrar tanto mi memoria como mi presentación.

---

## 📂 Contenidos del Proyecto

### 1. Data Engineering & Scraping

Procesamiento del dataset original **PhiUSIIL Phishing URL Dataset**, disponible en el  
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset).

Este dataset de grandes dimensiones (~235.000 instancias) ha sido el núcleo del trabajo.  
Se desarrollaron scripts específicos para:

- **Extracción de activos:**  
  Descarga masiva y concurrente de favicons asociados a las URLs.

- **Curación de datos:**  
  Limpieza de rutas locales, gestión de valores nulos y normalización de imágenes.

- **Feature Engineering:**  
  Generación de nuevas variables heurísticas a partir de la estructura de las URLs y del contenido HTML.

---

### 2. Análisis de Dimensionalidad

Validación de la calidad de los datos mediante técnicas avanzadas de reducción de dimensiones:

- **PCA (Principal Component Analysis):**  
  Reducción lineal para determinar la varianza explicada por los componentes principales.

- **t-SNE:**  
  Visualización no lineal para identificar agrupamientos (*clusters*) de clases en el espacio latente.

---

### 3. Modelado Predictivo

Comparativa de diversas arquitecturas para la clasificación binaria de URLs:

- **Random Forest:**  
  Análisis de importancia de variables y detección/eliminación de *data leakage*.

- **ANN (Artificial Neural Networks):**  
  Redes densas con regularización mediante *Dropout*.

- **CNN (Convolutional Neural Networks):**  
  Arquitectura personalizada para el procesamiento de imágenes de favicons.

- **Transfer Learning (ResNet50):**  
  Uso de redes preentrenadas en ImageNet para maximizar la precisión del modelo.

---

### 4. Generación Sintética (GANs)

Investigación en arquitecturas generativas con el objetivo de combatir el desbalance de clases y robustecer el discriminador:

- **ncGAN (Non-Conditional GAN):**  
  Generación de muestras sin control explícito por etiquetas de clase.

- **cGAN (Conditional GAN):**  
  Generación de muestras condicionadas por la clase objetivo.

---

### 5. Línea Futura

Exploración preliminar de una posible línea de trabajo futura:

- **Autoencoder + GMM:**  
  Modelado del espacio latente mediante Mezclas Gaussianas para una generación más estable y realista.

---

## 🛠️ Instalación

1. **Clonar el repositorio:**

2. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```
---

## 💻 Requisitos del Sistema

- **Python 3.8 o superior**
- **GPU recomendada**  
  Necesaria para tiempos de entrenamiento óptimos en los modelos basados en ResNet y GANs.

---

## 👤 Autores

- **Autor principal:** Ángel Truque Contreras  
- **Director:** Javier Vales Alonso

---

**Mayores contribuyentes por y para siempre: Bicho y Comino**

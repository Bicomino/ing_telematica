import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
DATA_PATH = Path("data/raw/dataset.csv")  # Cambia por tu ruta real
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Columnas identificadas con fuga de información (Leakage)
COLS_TO_DROP = [
    'FILENAME', 'URLSimilarityIndex', 'HasSocialNet', 
    'HasCopyrightInfo', 'HasDescription', 'DomainTitleMatchScore'
]

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================
def extract_url_features(url):
    return {
        'url_length': len(str(url)),
        'num_dots': str(url).count('.'),
        'num_digits': sum(c.isdigit() for c in str(url)),
        'has_ip': int(bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", str(url)))),
        'has_at_symbol': int('@' in str(url)),
        'has_https': int(str(url).lower().startswith('https')),
        'num_special_chars': len(re.findall(r"[^\w\s]", str(url))),
        'has_suspicious_words': int(any(w in str(url).lower() for w in ['login', 'verify', 'bank', 'secure', 'account', 'update']))
    }

def extract_domain_features(domain):
    d = str(domain)
    return {
        'domain_length': len(d),
        'num_subdomains': max(0, d.count('.') - 1)
    }

def extract_title_features(title):
    t = str(title).lower() if isinstance(title, str) else ""
    return {
        'title_length': len(t),
        'has_login_word_in_title': int('login' in t),
        'has_secure_word_in_title': int('secure' in t)
    }

# ==========================================
# 3. CARGA Y PREPROCESAMIENTO
# ==========================================
def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    # Limpieza inicial de columnas innecesarias o con leakage
    df_clean = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    
    print("Extrayendo nuevas características...")
    url_feats = df_clean['URL'].apply(extract_url_features).apply(pd.Series)
    dom_feats = df_clean['Domain'].apply(extract_domain_features).apply(pd.Series)
    tit_feats = df_clean['Title'].apply(extract_title_features).apply(pd.Series)
    
    # Combinar y eliminar originales
    df_final = pd.concat([
        df_clean.drop(columns=['URL', 'Domain', 'TLD', 'Title'], errors='ignore'),
        url_feats, dom_feats, tit_feats
    ], axis=1)
    
    return df_final.dropna().drop_duplicates()

# ==========================================
# 4. ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
def train_discriminator(df):
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Validación Cruzada
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return clf, X_train, X_test, y_train, y_test

# ==========================================
# 5. VISUALIZACIÓN (PCA & t-SNE)
# ==========================================
def visualize_clusters(X_test, y_pred):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # PCA para reducción inicial (90% varianza)
    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE para visualización 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred, 
        palette='viridis', alpha=0.6, edgecolor='k'
    )
    plt.title('Visualización t-SNE de las Predicciones')
    plt.show()

# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Procesar datos
    if DATA_PATH.exists():
        data = load_and_preprocess(DATA_PATH)
        
        # 2. Entrenar
        model, X_train, X_test, y_train, y_test = train_discriminator(data)
        
        # 3. Evaluar
        y_pred = model.predict(X_test)
        print("\n--- REPORTE DE CLASIFICACIÓN ---")
        print(classification_report(y_test, y_pred))
        
        # 4. Guardar
        joblib.dump(model, MODEL_DIR / "discriminador_rf.pkl")
        joblib.dump(X_train.columns.tolist(), MODEL_DIR / "features_list.pkl")
        print(f"Modelo guardado en {MODEL_DIR}")
        
        # 5. Visualizar
        visualize_clusters(X_test, y_pred)
    else:
        print(f"Archivo no encontrado en {DATA_PATH}")
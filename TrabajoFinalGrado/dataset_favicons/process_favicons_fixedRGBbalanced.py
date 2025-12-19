#!/usr/bin/env python3
"""
process_favicons_fixedRGBbalanced.py

Versión robusta y equilibrada:
- Mantiene las imágenes a color (RGB)
- Maneja .png, .jpg, .ico, .svg (opcional) y .gif (primer frame)
- Balancea ambas clases a TARGET_PER_CLASS
"""

import os
import io
import cv2
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
import warnings

#---CONFIGURACIÓN--
CSV_PATH = Path(r"data/temp/favicons_results.csv")
OUTPUT_BASE = Path(r"data/processed/favicons_color_balanced_fixed")
IMG_SIZE = (64, 64)
LOG_FILE = OUTPUT_BASE / "process_errors.log"

# Número objetivo por clase
TARGET_PER_CLASS = 1595
RANDOM_STATE = 42 # reproducibilidad

#---SILENCIAR WARNINGS--
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

#---INTENTAR CARGAR CairoSVG (para SVG opcional)--
try:
    import cairosvg
    SVG_ENABLED = True
except Exception:
    SVG_ENABLED = False

#---CREAR DIRECTORIOS--
LEGIT_DIR = OUTPUT_BASE / "legit"
PHISH_DIR = OUTPUT_BASE / "phishing"
LEGIT_DIR.mkdir(parents=True, exist_ok=True)
PHISH_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

#---LEER CSV--
print(f" Leyendo CSV: {CSV_PATH}")
if not CSV_PATH.exists():
    print(f"Error: No se encuentra {CSV_PATH}")
    exit()

df = pd.read_csv(CSV_PATH)
df["label"] = df["label"].astype(int)
df = df[df["status"] == "downloaded"]
print(f"Encontradas {len(df)} imágenes descargadas correctamente.")

#---BALANCEAR CLASES--
df_legit = df[df["label"] == 1].copy()
df_phish = df[df["label"] == 0].copy()

n_legit = len(df_legit)
n_phish = len(df_phish)
print(f"- legítimas encontradas: {n_legit}")
print(f"- phishing encontradas: {n_phish}")

# Igualar ambas clases a TARGET_PER_CLASS
target = min(TARGET_PER_CLASS, n_legit, n_phish)
df_legit_sampled = df_legit.sample(n=target, random_state=RANDOM_STATE)
df_phish_sampled = df_phish.sample(n=target, random_state=RANDOM_STATE)

df_final = pd.concat([df_legit_sampled, df_phish_sampled], ignore_index=True)
df_final = df_final.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Total a procesar tras balanceo: {len(df_final)} (legit: {len(df_legit_sampled)}, phish: {len(df_phish_sampled)})")

#---FUNCIONES AUXILIARES--
def log_error(path, msg):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"{path}: {msg}\n")

def convert_svg(svg_path: Path) -> Path:
    if not SVG_ENABLED:
        return None
    tmp_path = svg_path.with_suffix(".converted.png")
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(tmp_path))
        return tmp_path
    except Exception as e:
        log_error(svg_path, f"Error convirtiendo SVG: {e}")
        return None

def process_gif(path: Path):
    """Carga el primer frame de un GIF y devuelve imagen BGR."""
    try:
        pil_img = Image.open(path)
        pil_img.seek(0)
        frame = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        log_error(path, f"Error procesando GIF: {e}")
        return None

#---PROCESAMIENTO PRINCIPAL--
processed, failed = 0, 0

for i, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Procesando imágenes"):
    path = Path(row["favicon_path"])
    label = int(row["label"])

    if not path.exists() or path.stat().st_size == 0:
        log_error(path, "Archivo inexistente o vacío")
        failed += 1
        continue

    #---Manejo de formatos--
    ext = path.suffix.lower()
    img = None

    if ext == ".svg":
        new_path = convert_svg(path)
        if not new_path or not new_path.exists():
            failed += 1
            continue
        path = new_path
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif ext == ".gif":
        img = process_gif(path)
        if img is None:
            failed += 1
            continue
    elif ext == ".ico":
        try:
            pil_img = Image.open(path).convert("RGBA")
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            log_error(path, f"Error leyendo ICO: {e}")
            failed += 1
            continue
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None or img.size == 0:
        log_error(path, "OpenCV no pudo leer la imagen")
        failed += 1
        continue

    #---Normalizar canales--
    try:
        if len(img.shape) == 3:
            ch = img.shape[-1]
            if ch == 4: # Transparencia -> fondo blanco
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3]
                img = (255 * (1 - alpha[..., None]) + rgb * alpha[..., None]).astype("uint8")
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            log_error(path, f"Formato no reconocido: shape {img.shape}")
            failed += 1
            continue
    except Exception as e:
        log_error(path, f"Error normalizando canales: {e}")
        failed += 1
        continue

    #---Redimensionar--
    try:
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    except Exception as e:
        log_error(path, f"Error redimensionando: {e}")
        failed += 1
        continue

    #---Guardar--
    out_dir = LEGIT_DIR if label == 1 else PHISH_DIR
    out_path = out_dir / f"{i:06d}_{path.stem}.png"

    try:
        cv2.imwrite(str(out_path), img)
        processed += 1
    except Exception as e:
        log_error(path, f"Error guardando: {e}")
        failed += 1
        continue

print("\n Conversión completada.")
print(f"- {processed} imágenes procesadas correctamente.")
print(f"- {failed} fallidas (ver {LOG_FILE})")
print(f"- SVG activado: {SVG_ENABLED}")
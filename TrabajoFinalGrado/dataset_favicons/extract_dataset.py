#!/usr/bin/env python3
"""
extract_dataset.py

Descarga favicons de URLs con HasFavicon == 1 del CSV.
"""

import os
import csv
import time
import re
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

#----------CONFIGURACIÓN---------
CSV_PATH = r"data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_DIR = r"data/temp/favicons_descargados"
RESULT_CSV = os.path.join(OUTPUT_DIR, "favicons_results.csv")
MAX_WORKERS = 24 # Aumenta si tu CPU lo permite
REQUEST_TIMEOUT = 8 # Timeout más corto para evitar bloqueos
SAVE_EVERY = 1000 # Guardar progreso cada N URLs
USER_AGENT = "Mozilla/5.0 (compatible; favicon-downloader/1.1; +https://example.local/)"
SLEEP_BETWEEN = 0.01 # Pequeña pausa entre tareas

#----------FUNCIONES AUXILIARES---------
def sanitize_filename(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_.-]', '_', s)
    return s[:200]

def get_domain_base(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or parsed.path
        return f"{scheme}://{netloc}"
    except Exception:
        return None

def fetch_html(url: str):
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.text, r.url
    except Exception:
        return None, None

def find_icon_links_from_html(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all("link", rel=lambda v: v and any(x in v.lower() for x in ["icon", "shortcut", "apple-touch-icon"]))
    results = []
    for t in tags:
        href = t.get("href")
        if href:
            results.append(urljoin(base_url, href))
    # quitar duplicados manteniendo orden
    seen, dedup = set(), []
    for u in results:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup

def download_binary(url: str):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")

#----------LÓGICA DE PROCESAMIENTO---------
def process_url(index: int, url: str, label: str):
    out = {
        "URL": url,
        "label": label,
        "favicon_path": "",
        "favicon_url": "",
        "status": "not_found",
        "error": ""
    }
    try:
        base = get_domain_base(url)
        if not base:
            out["status"] = "bad_url"
            out["error"] = "No se pudo parsear la URL"
            return out

        html, final_page = fetch_html(url)
        if html is None:
            html, final_page = fetch_html(base)

        candidates = []
        if html:
            candidates.extend(find_icon_links_from_html(html, final_page or base))
        
        candidates.append(urljoin(base, "/favicon.ico"))

        last_err = "Sin favicon válido"
        for cand in candidates:
            try:
                data, ctype = download_binary(cand)
                if not data:
                    continue

                # Extensión según tipo
                if "png" in ctype:
                    ext = ".png"
                elif "svg" in ctype:
                    ext = ".svg"
                elif "jpeg" in ctype or "jpg" in ctype:
                    ext = ".jpg"
                elif "ico" in ctype or cand.lower().endswith(".ico"):
                    ext = ".ico"
                else:
                    ext = ".bin"

                domain = sanitize_filename(urlparse(base).netloc)
                fname = f"{index:06d}_{domain}{ext}"
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                path = os.path.join(OUTPUT_DIR, fname)
                with open(path, "wb") as f:
                    f.write(data)

                out.update({
                    "favicon_path": path,
                    "favicon_url": cand,
                    "status": "downloaded"
                })
                return out

            except Exception as e_cand:
                last_err = str(e_cand)
                continue

        out["status"] = "not_downloaded"
        out["error"] = last_err
        return out

    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out

#----------FUNCIÓN PRINCIPAL---------
def main():
    print(" leyendo CSV:", CSV_PATH)
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encuentra el archivo {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, usecols=lambda c: c.lower() in ["url", "hasfavicon", "label"])
    df.columns = [c.lower() for c in df.columns]
    
    df = df[df["hasfavicon"] == 1].reset_index(drop=True)
    print(f"Encontradas {len(df)} URLs con HasFavicon == 1")

    # Cargar progreso previo si existe
    results = []
    processed_urls = set()
    if os.path.exists(RESULT_CSV):
        print(f" Reanudando desde: {RESULT_CSV}")
        prev = pd.read_csv(RESULT_CSV)
        processed_urls = set(prev["URL"])
        results = prev.to_dict("records")
        df = df[~df["url"].isin(processed_urls)]
        print(f"Se omitirán {len(processed_urls)} URLs ya procesadas. Restan {len(df)}.")

    # Descarga concurrente
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_url, i, row["url"], row["label"]): row["url"] for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Descargando favicons"):
            try:
                res = fut.result()
            except Exception as e:
                res = {"URL": futures[fut], "label": "", "favicon_path": "", "favicon_url": "", "status": "error", "error": str(e)}
            
            results.append(res)

            # Guardar progreso cada N resultados
            if len(results) % SAVE_EVERY == 0:
                pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

    # Guardado final
    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
    print(" Descarga completa. Resultados guardados en:", RESULT_CSV)

#----------EJECUCIÓN---------
if __name__ == "__main__":
    main()
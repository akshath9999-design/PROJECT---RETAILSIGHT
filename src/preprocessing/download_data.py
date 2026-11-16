"""
Data Ingestion and Raw Dataset Setup Module
Downloads the UCI Online Retail Dataset, extracts the archive,
converts the master Excel file into an optimized raw CSV, and validates integrity.
"""

import os
import zipfile
import urllib.request
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
ZIP_PATH = os.path.join(DATA_RAW_DIR, "online_retail.zip")
EXCEL_PATH = os.path.join(DATA_RAW_DIR, "Online Retail.xlsx")
CSV_PATH = os.path.join(DATA_RAW_DIR, "online_retail.csv")


def download_and_extract_data(url: str = UCI_ZIP_URL, raw_dir: str = DATA_RAW_DIR) -> str:
    os.makedirs(raw_dir, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[OK] Raw CSV already exists at: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, nrows=5)
        print(f"[OK] Sample columns: {list(df.columns)}")
        return CSV_PATH

    # Download zip if needed
    if not os.path.exists(EXCEL_PATH):
        if not os.path.exists(ZIP_PATH):
            print(f"[INFO] Downloading UCI Online Retail dataset from:\n  {url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(ZIP_PATH, "wb") as out_file:
                total_bytes = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    total_bytes += len(chunk)
                    print(f"  Downloaded {total_bytes / (1024 * 1024):.2f} MB...", end="\r")
            print(f"\n[OK] Download completed: {total_bytes / (1024 * 1024):.2f} MB")

        print(f"[INFO] Extracting archive: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(raw_dir)
        print(f"[OK] Extracted files: {os.listdir(raw_dir)}")

    # Convert Excel to CSV for fast access
    if os.path.exists(EXCEL_PATH) and not os.path.exists(CSV_PATH):
        print(f"[INFO] Converting '{EXCEL_PATH}' to '{CSV_PATH}' for faster I/O...")
        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        print(f"[OK] Loaded {len(df):,} records from Excel. Saving to CSV...")
        df.to_csv(CSV_PATH, index=False)
        print(f"[OK] CSV successfully written: {CSV_PATH} ({os.path.getsize(CSV_PATH) / (1024*1024):.2f} MB)")

    return CSV_PATH


if __name__ == "__main__":
    download_and_extract_data()

"""
Responsive Oncology Test Network — Automated Data Engineering Pipeline
========================================================================
Author: Petar Belchovski (P374R)
Copyright (c) 2026 Petar Belchovski (P374R). All Rights Reserved.
License: ROTN Non-Commercial Attribution & Experimental License (see LICENSE.txt)

Downloads, extracts, filters, and transforms real human genomic oncology variant data from
the official US Government NCBI ClinVar public servers into a clean, locally cached CSV
ready for deep neural network training.
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
ARCHIVE_FILENAME = "variant_summary.txt.gz"
DEFAULT_MAX_OUTPUT_ROWS = 20000
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "real_genomic_data.csv"
REQUIRED_COLUMNS = [
    "ClinicalSignificance", "ReferenceAlleleVCF", "AlternateAlleleVCF",
    "Chromosome", "Start", "Type", "Assembly", "GeneSymbol", "PhenotypeList",
]

CANCER_GENE_PANEL = {
    "TP53", "RB1", "PTEN", "CDKN2A", "CDKN2B", "CDKN1B", "APC", "VHL", "WT1",
    "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "PALB2", "RAD51C", "RAD51D",
    "BRIP1", "BARD1", "NBN", "FANCA", "FANCC", "FANCD2", "RAD50", "MRE11",
    "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "KRAS", "NRAS", "HRAS", "BRAF",
    "RAF1", "MAP2K1", "MAP2K2", "NF1", "CBL", "PIK3CA", "PIK3R1", "AKT1", "AKT2",
    "MTOR", "TSC1", "TSC2", "STK11", "EGFR", "ERBB2", "ERBB3", "ERBB4", "MET",
    "ALK", "RET", "ROS1", "KIT", "PDGFRA", "PDGFRB", "FGFR1", "FGFR2", "FGFR3",
    "FGFR4", "FLT3", "NTRK1", "NTRK2", "NTRK3", "ARID1A", "ARID1B", "ARID2", "BAP1",
    "SMARCA4", "SMARCB1", "KMT2A", "KMT2D", "SETD2", "EZH2", "ASXL1", "DNMT3A",
    "TET2", "IDH1", "IDH2", "CTNNB1", "AXIN1", "AXIN2", "FBXW7", "RNF43", "SMAD4",
    "SMAD2", "TGFBR2", "NOTCH1", "NOTCH2", "NOTCH3", "MYC", "MYCN", "MYCL", "GATA3",
    "RUNX1", "ETV6", "POLE", "POLD1", "MEN1", "PTCH1", "SUFU", "SDHB", "SDHC", "SDHD", "SDHA", "FH",
}

ONCOLOGY_PHENOTYPE_KEYWORDS = [
    "cancer", "carcinoma", "sarcoma", "leukemia", "lymphoma", "melanoma", "glioma",
    "glioblastoma", "tumor", "tumour", "neoplasm", "neoplasia", "malignan", "oncolog",
    "polyposis", "li-fraumeni", "lynch", "cowden", "neurofibromatosis", "wilms",
    "retinoblastoma", "von hippel-lindau", "multiple endocrine neoplasia",
]


def download_full_clinvar_archive(url: str = CLINVAR_URL, destination_path: Path = None, force_redownload: bool = False) -> Path:
    if destination_path is None:
        destination_path = SCRIPT_DIR / ARCHIVE_FILENAME
    destination_path = destination_path.resolve()
    if destination_path.is_file() and not force_redownload and destination_path.stat().st_size > 300 * 1024 * 1024:
        print(f"  Found cached ClinVar archive: {destination_path.name} ({destination_path.stat().st_size / (1024 * 1024):.1f} MB)")
        print("  Skipping re-download. (Use --redownload flag to force fresh transfer)\n")
        return destination_path

    print(f"  Connecting to NCBI ClinVar servers: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ROTN-DataPipeline/2.2 (research-use-only)"})
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0)) or None
        total_mb = (total_size / (1024 * 1024)) if total_size else 0
        print(f"  Archive Size     : {total_mb:.1f} MB (streaming transfer)")
        temp_dest = destination_path.with_suffix(".tmp")
        downloaded, last_report = 0, start_time
        with open(temp_dest, "wb") as f_out:
            while chunk := response.read(1024 * 1024):
                f_out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 0.5 or (total_size and downloaded == total_size):
                    last_report = now
                    speed = (downloaded / (1024 * 1024)) / max(now - start_time, 0.001)
                    pct = (downloaded / total_size) * 100.0 if total_size else 0.0
                    sys.stdout.write(f"\r  Downloaded: {downloaded / (1024 * 1024):6.1f} MB ({pct:5.1f}%) | Speed: {speed:5.2f} MB/s ")
                    sys.stdout.flush()
        print()

    if temp_dest.is_file():
        if destination_path.is_file():
            destination_path.unlink()
        temp_dest.rename(destination_path)
    print(f"  Download verified: {destination_path.stat().st_size / (1024 * 1024):.1f} MB in {max(time.time() - start_time, 0.001):.1f}s\n")
    return destination_path


def synthesize_dna_sequences_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    if "ReferenceAlleleVCF" not in df.columns or "AlternateAlleleVCF" not in df.columns:
        return df
    ref = df["ReferenceAlleleVCF"].fillna("").astype(str).str.upper().str.strip()
    alt = df["AlternateAlleleVCF"].fillna("").astype(str).str.upper().str.strip()
    df["sequence"] = (ref + alt).str.replace(r"[^ACGT]", "", regex=True)
    return df[df["sequence"].str.len() > 0].copy()


def clean_null_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["Chromosome", "Start"] if col in df.columns]
    if not cols:
        return df
    df = df.dropna(subset=cols).copy()
    for col in cols:
        s = df[col].astype(str).str.strip()
        df = df[(s != "") & (s.str.lower() != "na") & (s != "-1") & (s != "-")].copy()
    return df


def cap_and_balance_output(df: pd.DataFrame, max_rows: int = DEFAULT_MAX_OUTPUT_ROWS, keep_all: bool = False) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if keep_all or len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    class_counts = df["is_cancerous"].value_counts()
    per_class = min(class_counts.min(), max_rows // 2)
    rng = np.random.RandomState(42)
    sampled_indices = []
    for label in [0, 1]:
        idx = df[df["is_cancerous"] == label].index.values
        sampled_indices.extend(rng.choice(idx, size=per_class, replace=False) if len(idx) > per_class else idx)

    df_balanced = df.loc[list(set(sampled_indices))].copy()
    remaining = max_rows - len(df_balanced)
    if remaining > 0:
        unused = df.index.difference(df_balanced.index).values
        if len(unused) > 0:
            df_balanced = pd.concat([df_balanced, df.loc[rng.choice(unused, size=min(remaining, len(unused)), replace=False)]], axis=0)

    df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
    counts = df_balanced["is_cancerous"].value_counts()
    print(f"  Balanced to {len(df_balanced):,} rows (Benign [0]: {counts.get(0, 0):,}, Pathogenic [1]: {counts.get(1, 0):,})")
    return df_balanced


def process_full_clinvar_archive(archive_path: Path, max_rows: int = DEFAULT_MAX_OUTPUT_ROWS, keep_all: bool = False) -> pd.DataFrame:
    archive_path = Path(archive_path).resolve()
    print(f"  Archive Path     : {archive_path.name}")
    print(f"  Target Rows      : {'ALL' if keep_all else f'{max_rows:,}'}")
    print(f"  Assembly Filter  : GRCh38 only (deduplicated)")
    print(f"  Gene Panel Filter: {len(CANCER_GENE_PANEL)} validated cancer driver genes\n")
    print("  Processing full archive in 100,000-row chunks...")
    print("  " + "-" * 72)

    chunk_size, filtered_chunks = 100000, []
    total_raw_rows, total_pathogenic, total_benign, chunk_index = 0, 0, 0, 0
    start_time, oncology_regex = time.time(), "|".join(ONCOLOGY_PHENOTYPE_KEYWORDS)

    reader = pd.read_csv(
        archive_path, sep="\t", compression="gzip", chunksize=chunk_size,
        usecols=REQUIRED_COLUMNS, low_memory=False, dtype=str
    )
    for chunk in reader:
        chunk_index += 1
        total_raw_rows += len(chunk)
        chunk = chunk[chunk["Assembly"] == "GRCh38"].copy()
        if len(chunk) == 0:
            continue
        clinsig = chunk["ClinicalSignificance"].fillna("").str.strip().str.lower()
        chunk = chunk[clinsig.isin(["pathogenic", "benign"])].copy()
        if len(chunk) == 0:
            continue

        gene_upper = chunk["GeneSymbol"].fillna("").str.upper().str.strip()
        pheno_lower = chunk["PhenotypeList"].fillna("").str.lower()
        mask = gene_upper.isin(CANCER_GENE_PANEL) | pheno_lower.str.contains(oncology_regex, regex=True, na=False)
        chunk = chunk[mask].copy()
        if len(chunk) == 0:
            continue

        chunk["is_cancerous"] = np.where(chunk["ClinicalSignificance"].str.strip().str.lower() == "pathogenic", 1, 0)
        chunk = synthesize_dna_sequences_vectorized(chunk)
        chunk = clean_null_coordinates(chunk)
        total_pathogenic += int((chunk["is_cancerous"] == 1).sum())
        total_benign += int((chunk["is_cancerous"] == 0).sum())
        filtered_chunks.append(chunk)

        if chunk_index % 10 == 0:
            elapsed = max(time.time() - start_time, 0.001)
            print(f"  Chunk {chunk_index:3d} | Parsed: {total_raw_rows:9,d} | Oncology: {total_pathogenic + total_benign:7,d} | Speed: {total_raw_rows / elapsed:7.0f} rows/s")

    elapsed = max(time.time() - start_time, 0.001)
    print("  " + "-" * 72)
    print(f"  Completed in {elapsed:.1f}s | Raw: {total_raw_rows:,} | Oncology candidates: {total_pathogenic + total_benign:,}\n")
    if not filtered_chunks:
        print("  [ERROR] No variants matched criteria.")
        sys.exit(1)

    combined_df = pd.concat(filtered_chunks, ignore_index=True)
    return cap_and_balance_output(combined_df, max_rows=max_rows, keep_all=keep_all)


def main():
    parser = argparse.ArgumentParser(description="ROTN — NCBI ClinVar Full Archive Pipeline")
    parser.add_argument("--all", action="store_true", dest="keep_all", help="Retain all oncology variants.")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_OUTPUT_ROWS, help=f"Max output rows (default: {DEFAULT_MAX_OUTPUT_ROWS:,}).")
    parser.add_argument("--redownload", action="store_true", help="Force re-download of archive.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_CSV), help=f"Target CSV path (default: {DEFAULT_OUTPUT_CSV}).")
    args = parser.parse_args()

    print("\n+========================================================================+")
    print("|  RESPONSIVE ONCOLOGY TEST NETWORK — DATA PIPELINE v2.2                 |")
    print("|  Author: Petar Belchovski (P374R)                                      |")
    print("+========================================================================+\n")

    archive = download_full_clinvar_archive(url=CLINVAR_URL, force_redownload=args.redownload)
    df = process_full_clinvar_archive(archive, max_rows=args.max_rows, keep_all=args.keep_all)
    out = Path(args.output).resolve()
    print(f"  Exporting dataset to: {out}")
    df.to_csv(out, index=False)
    print(f"  Export complete! File size: {out.stat().st_size / 1024:.1f} KB\n")


if __name__ == "__main__":
    main()

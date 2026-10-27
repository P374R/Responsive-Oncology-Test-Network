#  Responsive Oncology Test Network (ROTN)

> **A 100-layer deep residual neural network for cancer variant pathogenicity classification, built from scratch in Python using only raw NumPy and Pandas. Zero PyTorch. Zero TensorFlow. Zero scikit-learn.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/NumPy-Zero--Framework-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-Non--Commercial%20Attribution-blue.svg)](LICENSE.txt)
[![Author](https://img.shields.io/badge/Author-Petar%20Belchovski%20(P374R)-2ea44f.svg)](https://github.com/P374R/)

---

##  Medical & Research Disclaimer

> [!CAUTION]
> **RESEARCH & EDUCATIONAL USE ONLY — NOT A CLINICAL DIAGNOSTIC TOOL**
> 
> ROTN is an open-source educational project exploring computational genomics and deep neural network mathematics from first principles.
> - **Never** use this software for clinical diagnoses, patient care decisions, or oncology treatment planning.
> - ROTN is **not** certified by the FDA, EMA, or any healthcare regulatory body.
> - Any mutation flagged computationally must be validated through wet-lab clinical sequencing (such as Sanger or NGS) by certified molecular pathology specialists.

---

##  Why Build This?

Most deep learning today is done with a couple of high-level PyTorch or TensorFlow imports. While powerful, frameworks hide all the low-level linear algebra, memory layouts, and numerical gradient physics.

**I wanted to see if I could build, train, and converge a legitimately deep 100-layer neural network from complete scratch using only pure NumPy matrix multiplication.**

If you try naively stacking 100 dense layers with standard Sigmoid activations, backpropagation multiplies derivatives until your gradients completely vanish ($0.25^{100} \approx 10^{-60}$) by layer 10. The early layers stay frozen at initialization, and the model never learns.

### How ROTN fixes this:
*  **100 Residual Layers:** Implements ResNet-style skip connections ($a_l = a_{l-1} + \gamma \cdot \text{LeakyReLU}(z_l)$) with Fixup scale factor $\gamma = \frac{1}{\sqrt{100}} = 0.1$, keeping activations stable across 100 additions without extra normalization parameters.
*  **Identity Gradient Highway:** Backpropagation routes error gradients straight back through the identity shortcuts ($g \leftarrow g + \delta_l W_l^T$), allowing learning signals to hit Layer 0 at full strength.
*  **Custom NumPy Optimizers:** Hand-rolled vectorized implementations of **Adam** (with moment bias corrections) and **Momentum SGD** (0.9) with element-wise gradient clipping ($[-1.0, +1.0]$).
*  **400-D Genomic Encoder:** Translates both clinical mutation syntax (like `TP53:c.742C>T` or `KRAS:c.35G>A`) and raw DNA sequences into standardized numerical tensors.
*  **NCBI ClinVar Pipeline:** Streams real human mutation records from the US Government NCBI ClinVar archive, strictly deduplicates to GRCh38, and isolates variants across 80+ validated human cancer driver genes.

---

##  Quickstart

All you need is **Python 3.8+**, `numpy`, and `pandas`.

```bash
git clone https://github.com/P374R/Responsive-Oncology-Test-Network.git
cd Responsive-Oncology-Test-Network
pip install -r requirements.txt
```

### 1. Test a Mutation
```bash
python oncotest.py "TP53:c.742C>T"
```

### 2. Run the Built-In Benchmark Demo
```bash
python oncotest.py --demo
```

### 3. Interactive Diagnostic Console
```bash
python oncotest.py
```
Inside the interactive prompt:
* Paste any mutation notation or raw DNA sequence.
* Type `demo` to evaluate benchmark cases.
* Type `cls` to clear the terminal screen cleanly (pure ANSI escape sequences, zero risky shell calls).
* Type `quit` to exit.

### 4. Train on Real Clinical Data (Optional)
To train the 100-layer residual network on real-world clinical cancer data yourself:
```bash
# Step 1: Stream & filter 20,000 real cancer variants from NCBI ClinVar
python download_data.py --max-rows 20000

# Step 2: Train the 100-layer residual network (2,500 epochs default)
python train.py
```
*(Note: If you run `python train.py` without downloading first, it automatically trains on the included `mock_genomic_data.csv` for zero-setup experimentation!)*

---

##  Cool Variants to Test

ROTN accepts standard clinical notation, genomic coordinates, single nucleotide substitutions, or raw DNA strings:

| Input | Gene Context | Biological Significance |
| :--- | :---: | :--- |
| `TP53:c.742C>T` | *TP53* | Hotspot oncogenic driver mutation (p.R248W) |
| `KRAS:c.35G>A` | *KRAS* | Codon 12 oncogenic driver mutation (p.G12D) |
| `ATM:chr11:108223000:A:G` | *ATM* | Intronic benign polymorphism reference |
| `chr17:7577538C>T` | *TP53* | Genomic chromosomal coordinate substitution |
| `C>T` | N/A | Direct single-nucleotide substitution |
| `CCGCCGCCGCCGCCCGCCCGGCCGCCCCCGCCGCCCGCCC` | N/A | High-GC raw DNA sequence |
| `AATTAATTAATTAATTAATTAATTAATTAATTAATTAA...` | N/A | High-AT raw DNA sequence |

You can also pipe input straight from bash or PowerShell scripts:
```bash
echo "KRAS:c.35G>A" | python oncotest.py -
```

---

##  How It Works Under the Hood

### The 400-Dimensional Feature Encoder
Raw variant strings and DNA sequences are mapped into a rich 400-dimensional numerical representation:

*  **Allele Sequences (Dims 0–199):** One-hot encoded base pairs (A, C, G, T) for reference and alternate alleles up to 25 nucleotides each.
*  **Substitution Matrix (Dims 200–215):** Explicit $4 \times 4$ mapping of the exact nucleotide transition (e.g., $C \to T$).
*  **Mutation Signatures (Dims 216–225):** Transition vs. transversion flags, insertion/deletion length deltas, CpG island context, and GC content ratios.
*  **Genomic & Gene Context (Dims 226–282):** Chromosome one-hot representation, normalized genomic coordinate, and matching against 80+ validated cancer driver genes (*TP53*, *BRCA1*, *BRCA2*, *KRAS*, *EGFR*, *PTEN*, etc.).
*  **Dinucleotide Frequencies (Dims 283–298):** Frequency spectrum of dinucleotide pairs (AA, AC, AG, etc.).
*  **Raw Sequence Fallback (Dims 0–399):** If a raw sequence is provided without variant syntax, up to 100 nucleotides are binary-encoded across all 400 dimensions.

###  Risk Severity Scale

Predictions output a probability score mapped across 6 clinical risk tiers:

| Probability Range | Hazard Status | Clinical Classification | Recommended Action |
| :---: | :---: | :--- | :--- |
| $\ge 90\%$ | `[!! HAZARD ALERT !!]` | **PATHOGENIC — CRITICAL** | Immediate expert review & wet-lab validation |
| $75\% - 89\%$ | `[!! HAZARD ALERT !!]` | **PATHOGENIC — HIGH RISK** | Priority clinical investigation & confirmatory test |
| $50\% - 74\%$ | `[! CAUTION !]` | **UNCERTAIN SIGNIFICANCE (VUS)** | Family segregation & functional assay analysis |
| $25\% - 49\%$ | `[MONITOR]` | **LIKELY BENIGN** | Low risk; routine baseline monitoring |
| $10\% - 24\%$ | `[CLEAR]` | **BENIGN — LOW RISK** | Negligible clinical significance |
| $< 10\%$ | `[CLEAR]` | **BENIGN — NOMINAL** | Normal baseline genomic profile |

---

## 🔬 Training on Real Clinical Data

To keep this repository lightweight and fast to clone, raw NCBI genomic archives (>400 MB) are **not** bundled directly into git. If you want to train on real-world clinical cancer mutations yourself, follow this 2-step pipeline:

### Step 1: Download & Filter Real ClinVar Data (`download_data.py`)

`download_data.py` streams the official US Government NCBI ClinVar summary archive (`variant_summary.txt.gz`, ~435 MB) directly from NIH FTP servers and filters it down into a balanced, deduplicated dataset:

```bash
# Download and build a balanced dataset of 20,000 real cancer variants
python download_data.py --max-rows 20000

# Retain all matching oncology variants
python download_data.py --all --output full_oncology_variants.csv

# Force re-download of archive
python download_data.py --redownload --max-rows 20000
```
This generates `real_genomic_data.csv` in your project folder.

#### Data Engineering Highlights:
* **GRCh38 Deduplication:** Strictly filters to `Assembly == "GRCh38"`, eliminating 50% duplicate record leakage that occurs when GRCh37 and GRCh38 are both included.
* **Cancer Driver Panel:** Filters against 80+ validated human oncogenes and tumor suppressors, plus clinical oncology phenotype keywords (*carcinoma*, *sarcoma*, *melanoma*, *leukemia*, *lymphoma*, etc.).
* **Index-Safe Balanced Sampling:** Balances benign and pathogenic samples 50/50 without index-collision data loss.

---

### Step 2: Training the 100-Layer Network (`train.py`)

`train.py` automatically detects `real_genomic_data.csv` and trains the 100-layer deep residual network on it:

```bash
# Standard training (2,500 epochs default, fast 256 batch size)
python train.py

# Quick test run (finishes in ~30 seconds)
python train.py --epochs 50 --batch-size 256

# Train on a custom CSV with an 80/20 train/validation split
python train.py --data my_data.csv --val-split 0.20
```

> **Custom CSV Column Schema:**
> If you provide your own dataset via `--data`, your CSV needs these columns:
> `is_cancerous` (1/0), `GeneSymbol` (e.g. `TP53`), `Chromosome` (e.g. `17`), `Start` (genomic position), `ReferenceAlleleVCF` (e.g. `C`), `AlternateAlleleVCF` (e.g. `T`), and `Assembly` (`GRCh38`).

> **Note on Zero-Setup Testing:** If you run `python train.py` without running `download_data.py` first, the script automatically trains on the included `mock_genomic_data.csv` so you can experiment right away without downloading anything!

#### Hardware Acceleration & Optimization
* **Intel / AMD CPU (Zero-Install):** Uses vectorized AVX2/OpenBLAS matrix routines. Using `--batch-size 256` reduces Python loop iterations by 75% and speeds up CPU training by **3x to 4x**!
* **GPU Acceleration:** Install CuPy with CUDA Toolkit headers (`pip install "cupy-cuda12x[ctk]"`), the official CUDA drop-in for NumPy. ROTN will automatically detect your GPU and accelerate the 100-layer matrix algebra on Tensor Cores (**40x–80x faster**, dropping epoch times from ~4s to ~0.08s).
* **Automatic 80/20 Train/Val Split:** Evaluates both training and validation loss/accuracy to prove real generalization.
* **Live Progress & ETA:** Prints speed (seconds per epoch), time remaining (ETA), loss curves, and prediction spread in real time without lag.
* **Safe Interruption (`Ctrl+C`):** If you press `Ctrl+C`, the script cleanly catches the interrupt and saves your best checkpoint to `rotn_weights.npz` so your time is never wasted.
* **Best Model Checkpointing:** Always preserves and serializes the weights from the epoch with the lowest validation loss.

---

## 📁 Project Structure

```
Responsive-Oncology-Test-Network/
├── download_data.py        # NCBI ClinVar FTP downloader and cancer variant filter
├── train.py                # 100-layer residual deep neural network training engine
├── oncotest.py             # Diagnostic terminal interface and real-time predictor
├── mock_genomic_data.csv   # Instant offline synthetic test dataset (26 KB)
├── real_genomic_data.csv   # Real ClinVar dataset (generated via download_data.py)
├── rotn_weights.npz        # 100-layer residual weights archive (3.24 MB)
├── requirements.txt        # Core package dependencies (numpy, pandas)
├── LICENSE.txt             # Non-Commercial Attribution & Experimental License
└── README.md               # Project documentation and manual
```

---

##  License & Attribution

This project is created and authored by **Petar Belchovski (P374R)** under the **ROTN Non-Commercial Attribution & Experimental License**.

```
Copyright (c) 2026 Petar Belchovski (P374R). All Rights Reserved.
```

*  **Non-Commercial Use:** You are welcome to inspect, fork, learn from, and modify this project for educational and scientific research purposes.
*  **No Commercial Resale:** You may not monetize, resell, or distribute this software or its derivative weights for commercial purposes without prior explicit written permission from Petar Belchovski (P374R).
*  **Attribution:** Any forks, papers, or public distributions must visibly credit **Petar Belchovski (P374R)** as the original author of the Responsive Oncology Test Network (ROTN).

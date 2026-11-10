# Responsive Oncology Test Network (ROTN) v1.1.0

### *Universal GPU Acceleration Engine Update*
**Author:** Petar Belchovski ([P374R](https://github.com/P374R))  
**License:** Non-Commercial Attribution & Experimental Research License  

---

## What is ROTN?
The **Responsive Oncology Test Network (ROTN)** is an open-source deep learning genomics suite designed to evaluate cancer variant pathogenicity from first principles. Built from scratch in Python using strictly **NumPy** and **Pandas** - zero PyTorch, zero TensorFlow, zero scikit-learn.

---

## What is New in v1.1.0 (Detailed Changelog)

Compared to the initial v1.0.0 release, version 1.1.0 brings significant hardware acceleration, memory optimization, and runtime workflow improvements:

| Feature / Area | v1.0.0 (Initial) | v1.1.0 (Current) | Benefit |
| :--- | :--- | :--- | :--- |
| **Hardware Engine** | CPU-only default with basic CuPy check | Universal Self-Contained GPU Engine (`--device auto`) | Auto-probes NVIDIA RTX/GTX & AMD Radeon hardware |
| **Device Fallback** | Manual configuration | Seamless automatic CPU fallback with hardware hints | Zero crashes if GPU is unavailable or unconfigured |
| **Floating-Point Precision** | `float64` (64-bit double precision) | `float32` (32-bit single precision) across all tensors | 50% lower VRAM, ~11x faster matmul on Tensor Cores |
| **Weight Archive Size** | 3.24 MB (`rotn_weights.npz`) | ~1.59 MB (`rotn_weights.npz`) | Lighter repository footprint and faster load times |
| **Default Batch Sizing** | Fixed batch size 256 | Adaptive: 4096 (GPU) / 256 (CPU) | Minimal Python loop iterations on GPU |
| **VRAM Strategy** | Batch-by-batch host memory slicing | Full in-device VRAM residency & GPU shuffling | Eliminates PCIe host-to-device bus transfer bottlenecks |
| **Windows Shebang Routing** | Standard python invocation | Embedded `#!python3.12` shebang line | Direct `py train.py` launches GPU environment automatically |
| **Default Training Budget** | 2,500 epochs | 1,000 epochs with adaptive logging intervals | Faster standard convergence cycle |
| **Documentation & Badges** | v1.0.0 baseline | v1.1.0 versioning, GPU badges, and hardware guides | Clear user guidance for NVIDIA CUDA and AMD ROCm |

---

## Key Highlights in v1.1.0

### 1. High-Saturation GPU Acceleration Engine (`train.py`) [NEW]
* **Full Device VRAM Residency:** Feature matrices (`X_train`, `y_train`) and validation tensors are pre-loaded directly into GPU device memory at startup with on-device random permutation shuffling (`cp.random.permutation`), eliminating PCIe bus transfer stalls and keeping GPU compute units continuously saturated.
* **Zero Extra Dependencies:** Self-contained entirely within `train.py` with zero auxiliary files or framework bloat.
* **ANY NVIDIA GPU Supported:** High-speed Tensor Core acceleration on all NVIDIA RTX (20/30/40/50 series) and GTX cards via CuPy CUDA (`pip install "cupy-cuda12x[ctk]"`).
* **ANY AMD GPU Supported:** Full acceleration across all AMD Radeon cards (RX 5000/6000/7000/8000 series, Vega, Instinct) via CuPy ROCm/HIP (`pip install cupy-rocm-6-0`).
* **Intelligent Auto-Detection:** Automatically detects physical graphics hardware and active runtimes, with seamless fallback to multi-threaded CPU OpenBLAS/AVX2.
* **Adaptive Hardware Batch Sizing:** Automatically chooses `--batch-size 4096` on GPU by default for peak Tensor Core throughput, and switches to safe `--batch-size 256` on CPU to protect against thermal throttling. Override anytime with `--batch-size <N>`.
* **Pure float32 Precision:** Weights, biases, and genomic tensors operate in IEEE 754 single precision (`float32`), cutting VRAM consumption by 50% (~1.59 MB vs 3.24 MB archive) and unlocking full Tensor Core FP32 compute lanes.
* **Active Status Reporting:** Cleanly reports exact GPU model name, total VRAM (GB), backend architecture, and residency status upon launch.

### 2. 100-Layer Deep Residual Neural Network (`train.py`)
* **First-Principles Deep Architecture:** 100 dense residual layers (437,569 parameters) implemented entirely with raw linear algebra and backpropagation.
* **Vanquishing the Vanishing Gradient Problem:** Standard deep feed-forward networks numerically collapse at 100 layers ($0.25^{100} \approx 10^{-60}$). ROTN introduces **Fixup residual scaling** ($\gamma = \frac{1}{\sqrt{100}} = 0.1$) and identity shortcut gradient highways ($g \leftarrow g + \delta_l W_l^T$), ensuring gradient signals reach Layer 0 at full strength.
* **Custom Hand-Rolled Optimizers:** Vectorized NumPy implementations of **Adam** (with bias-corrected first and second moments) and **Momentum SGD** (0.9) with element-wise gradient clipping.

### 3. Clinical Diagnostic Terminal Suite (`oncotest.py`)
* **Multi-Format Genomic Parser:** Accepts standard clinical HGVS notations (e.g. `TP53:c.742C>T`, `KRAS:c.35G>A`), chromosomal coordinates (`ATM:chr11:108223000:A:G`), or raw DNA nucleotide sequences.
* **ACMG/AMP Aligned 6-Tier Risk Stratification:** Outputs structured diagnostic reports mapping probabilities to clinical hazard statuses (`PATHOGENIC - CRITICAL`, `HIGH RISK`, `VUS`, `LIKELY BENIGN`, `BENIGN`).
* **Interoperable Pipeline CLI:** Supports direct CLI arguments, interactive console mode, and piped STDIN (`echo "KRAS:c.35G>A" | python oncotest.py -`).
* **Built-in Benchmark Demo:** Run `python oncotest.py --demo` to immediately evaluate benchmark oncology mutations.

### 4. NCBI ClinVar Automated Data Engine (`download_data.py`)
* **Streaming US Government Archive:** Streams official NCBI ClinVar records (`variant_summary.txt.gz`) directly from NIH servers.
* **GRCh38 Deduplication:** Strictly enforces `Assembly == "GRCh38"`, eliminating duplicate coordinate leakage.
* **Cancer Driver Panel:** Filters against 80+ validated human oncogenes and tumor suppressors (*TP53*, *KRAS*, *BRCA1*, *BRCA2*, *EGFR*, *PTEN*, etc.).
* **Balanced 50/50 Sampling:** Creates index-safe balanced sets of pathogenic and benign clinical mutations.

### 5. 400-Dimensional Biochemical Feature Encoding
* **Allele One-Hot Representations:** Captures reference and alternate alleles up to 25 bp each.
* **Transition vs. Transversion Matrices:** Models nucleotide mutational physics ($4 \times 4$ substitution space).
* **Biochemical Context Signatures:** Models CpG island context, insertion/deletion length deltas, GC content skew, and dinucleotide frequencies.

---

## Quickstart

```bash
# 1. Clone & install dependencies
git clone https://github.com/P374R/Responsive-Oncology-Test-Network.git
cd Responsive-Oncology-Test-Network
pip install -r requirements.txt

# 2. Test a cancer mutation immediately
python oncotest.py "TP53:c.742C>T"

# 3. Run benchmark demonstrations
python oncotest.py --demo

# 4. Train model (1,000 epochs default, auto GPU acceleration)
python train.py
```

---

## License & Attribution
Created by **Petar Belchovski (P374R)** under the **ROTN Non-Commercial Attribution & Experimental Research License**. Free to study, inspect, benchmark, and modify for non-commercial educational and scientific research purposes with proper attribution.

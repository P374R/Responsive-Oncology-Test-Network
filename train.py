"""
Responsive Oncology Test Network — Deep Residual Mini-Batch Training Engine
============================================================================
Author: Petar Belchovski (P374R)
Copyright (c) 2026 Petar Belchovski (P374R). All Rights Reserved.
License: ROTN Non-Commercial Attribution & Experimental License (see LICENSE.txt)

A zero-framework deep residual neural network for classifying human genomic
mutations as benign or pathogenic, built strictly with raw NumPy matrix linear
algebra and mini-batch optimization (Adam / SGD with Momentum).
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

# Hardware Acceleration Detection (CPU vs. NVIDIA CUDA GPU via CuPy)
try:
    # pyrefly: ignore[missing-import]
    import cupy as cp
    _test_gpu = cp.zeros(2) + 1
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
TOP_CANCER_GENES = [
    "TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "PTEN", "APC", "ATM",
    "BRAF", "PIK3CA", "RB1", "MYC", "ERBB2", "MLH1", "MSH2", "CDKN2A",
    "VHL", "RET", "ALK", "MET"
]
GENE_TO_INDEX = {g: i for i, g in enumerate(TOP_CANCER_GENES)}
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]
CHROM_TO_INDEX = {c: i for i, c in enumerate(CHROMOSOMES)}

NUCLEOTIDE_LUT = np.zeros((256, 4), dtype=np.float64)
NUCLEOTIDE_LUT[ord("A")] = NUCLEOTIDE_LUT[ord("a")] = [0.0, 0.0, 0.0, 1.0]
NUCLEOTIDE_LUT[ord("C")] = NUCLEOTIDE_LUT[ord("c")] = [0.0, 0.0, 1.0, 0.0]
NUCLEOTIDE_LUT[ord("G")] = NUCLEOTIDE_LUT[ord("g")] = [0.0, 1.0, 0.0, 0.0]
NUCLEOTIDE_LUT[ord("T")] = NUCLEOTIDE_LUT[ord("t")] = [1.0, 0.0, 0.0, 0.0]
BASE_PAIR_INDICES = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence_batch(sequences: list, max_len: int = 25) -> np.ndarray:
    n = len(sequences)
    if n == 0:
        return np.zeros((0, max_len * 4), dtype=np.float64)
    fixed = "".join(str(s)[:max_len].ljust(max_len) for s in sequences)
    b = np.frombuffer(fixed.encode("ascii", errors="replace"), dtype=np.uint8).reshape(n, max_len)
    return NUCLEOTIDE_LUT[b].reshape(n, max_len * 4)


def encode_dataset_to_400dim(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    features = np.zeros((n, 400), dtype=np.float64)
    if "ReferenceAlleleVCF" in df.columns and "AlternateAlleleVCF" in df.columns:
        ref_s = df["ReferenceAlleleVCF"].fillna("").astype(str).str.upper().str.strip()
        alt_s = df["AlternateAlleleVCF"].fillna("").astype(str).str.upper().str.strip()
        features[:, 0:100] = encode_sequence_batch(ref_s.tolist(), max_len=25)
        features[:, 100:200] = encode_sequence_batch(alt_s.tolist(), max_len=25)

        ref_len, alt_len = ref_s.str.len().values, alt_s.str.len().values
        is_snv = (ref_len == 1) & (alt_len == 1)
        ref_first, alt_first = ref_s.str[0].values, alt_s.str[0].values

        transitions = np.zeros(n, dtype=np.float64)
        transversions = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if is_snv[i]:
                r, a = ref_first[i], alt_first[i]
                if r in BASE_PAIR_INDICES and a in BASE_PAIR_INDICES:
                    features[i, 200 + BASE_PAIR_INDICES[r] * 4 + BASE_PAIR_INDICES[a]] = 1.0
                    if (r == "A" and a == "G") or (r == "G" and a == "A") or (r == "C" and a == "T") or (r == "T" and a == "C"):
                        transitions[i] = 1.0
                    else:
                        transversions[i] = 1.0

        features[:, 216], features[:, 217] = transitions, transversions
        features[:, 218] = (alt_len > ref_len).astype(np.float64)
        features[:, 219] = (ref_len > alt_len).astype(np.float64)
        len_delta = alt_len - ref_len
        features[:, 220] = np.tanh(len_delta / 10.0)
        features[:, 221] = np.minimum(np.abs(len_delta), 50.0) / 50.0
        features[:, 222] = ref_s.str.contains("CG", regex=False).astype(np.float64).values
        features[:, 223] = (((ref_first == "C") & (alt_first == "T")) | ((ref_first == "G") & (alt_first == "A"))).astype(np.float64)
        features[:, 224] = np.where(ref_len > 0, (ref_s.str.count("G").values + ref_s.str.count("C").values) / np.maximum(ref_len, 1), 0.0)
        features[:, 225] = np.where(alt_len > 0, (alt_s.str.count("G").values + alt_s.str.count("C").values) / np.maximum(alt_len, 1), 0.0)

        if "Chromosome" in df.columns:
            chroms = df["Chromosome"].astype(str).str.upper().str.replace("CHR", "").str.strip().values
            for i, c in enumerate(chroms):
                if c in CHROM_TO_INDEX:
                    features[i, 226 + CHROM_TO_INDEX[c]] = 1.0
        if "Start" in df.columns:
            starts = pd.to_numeric(df["Start"], errors="coerce").fillna(0).values
            features[:, 250] = np.clip(starts / 250000000.0, 0.0, 1.0)
        if "GeneSymbol" in df.columns:
            genes = df["GeneSymbol"].fillna("").astype(str).str.upper().str.strip().values
            for i, g in enumerate(genes):
                if g in GENE_TO_INDEX:
                    features[i, 251 + GENE_TO_INDEX[g]] = 1.0
                elif g:
                    h = hash(g)
                    for k in range(12):
                        features[i, 271 + k] = 1.0 if ((h >> (k * 2)) & 1) else 0.0

        concat_seq = ref_s + alt_s
        dinucs = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
        for idx, dinuc in enumerate(dinucs):
            features[:, 283 + idx] = np.clip(concat_seq.str.count(dinuc).values / 10.0, 0.0, 1.0)
    else:
        seqs = df["sequence"].astype(str).str.upper().str.strip().tolist()
        features = encode_sequence_batch(seqs, max_len=100)
    return features


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.maximum(z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def binary_cross_entropy_loss(predictions, targets, eps: float = 1e-15, xp=None) -> float:
    if xp is None:
        xp = cp if (CUPY_AVAILABLE and isinstance(predictions, cp.ndarray)) else np
    p = xp.clip(predictions, eps, 1.0 - eps)
    return float(-xp.mean(targets * xp.log(p) + (1.0 - targets) * xp.log(1.0 - p)))


class DeepGenomicNetwork:
    def __init__(
        self, input_size: int = 400, hidden_layers: int = 100, hidden_size: int = 64,
        output_size: int = 1, learning_rate: float = 0.001, optimizer: str = "adam", alpha: float = 0.01,
        device: str = "cpu",
    ):
        self.input_size, self.hidden_layers, self.hidden_size = input_size, hidden_layers, hidden_size
        self.output_size, self.learning_rate, self.alpha = output_size, learning_rate, alpha
        self.optimizer_type, self.gamma = optimizer.lower(), 1.0 / np.sqrt(hidden_layers)
        self.device = device.lower()
        self.xp = cp if (self.device == "gpu" and CUPY_AVAILABLE) else np

        self.weights, self.biases = [], []
        w0 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        b0 = np.zeros((1, hidden_size), dtype=np.float64)
        self.weights.append(self.xp.asarray(w0))
        self.biases.append(self.xp.asarray(b0))
        for _ in range(1, hidden_layers):
            wl = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
            bl = np.zeros((1, hidden_size), dtype=np.float64)
            self.weights.append(self.xp.asarray(wl))
            self.biases.append(self.xp.asarray(bl))
        w_out = np.random.randn(hidden_size, output_size) * (0.01 / np.sqrt(hidden_size))
        b_out = np.zeros((1, output_size), dtype=np.float64)
        self.weights.append(self.xp.asarray(w_out))
        self.biases.append(self.xp.asarray(b_out))

        self.step = 0
        if self.optimizer_type == "adam":
            self.m_w = [self.xp.zeros_like(w) for w in self.weights]
            self.v_w = [self.xp.zeros_like(w) for w in self.weights]
            self.m_b = [self.xp.zeros_like(b) for b in self.biases]
            self.v_b = [self.xp.zeros_like(b) for b in self.biases]
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        else:
            self.v_w = [self.xp.zeros_like(w) for w in self.weights]
            self.v_b = [self.xp.zeros_like(b) for b in self.biases]
            self.momentum = 0.9
        self._total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    @property
    def total_parameters(self) -> int:
        return self._total_params

    def forward(self, X) -> tuple:
        activations, pre_activations = [X], [None]
        z0 = X @ self.weights[0] + self.biases[0]
        cur = self.xp.maximum(z0, self.alpha * z0)
        activations.append(cur)
        pre_activations.append(z0)

        for l in range(1, self.hidden_layers):
            z = cur @ self.weights[l] + self.biases[l]
            cur = cur + self.gamma * self.xp.maximum(z, self.alpha * z)
            activations.append(cur)
            pre_activations.append(z)

        z_out = cur @ self.weights[-1] + self.biases[-1]
        p_out = 1.0 / (1.0 + self.xp.exp(-self.xp.clip(z_out, -500.0, 500.0)))
        activations.append(p_out)
        pre_activations.append(z_out)
        return activations, pre_activations

    def backpropagate(self, X, y, activations: list, pre_activations: list) -> float:
        self.step += 1
        batch_size = X.shape[0]
        predictions = activations[-1]
        loss = binary_cross_entropy_loss(predictions, y, xp=self.xp)

        d_out = (predictions - y) / batch_size
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        grad_w[-1] = activations[-2].T @ d_out
        grad_b[-1] = self.xp.sum(d_out, axis=0, keepdims=True)

        g = d_out @ self.weights[-1].T
        for l in range(self.hidden_layers - 1, 0, -1):
            delta_l = (self.gamma * g) * self.xp.where(pre_activations[l + 1] > 0, 1.0, self.alpha)
            grad_w[l] = activations[l].T @ delta_l
            grad_b[l] = self.xp.sum(delta_l, axis=0, keepdims=True)
            g = g + delta_l @ self.weights[l].T

        delta_0 = g * self.xp.where(pre_activations[1] > 0, 1.0, self.alpha)
        grad_w[0] = X.T @ delta_0
        grad_b[0] = self.xp.sum(delta_0, axis=0, keepdims=True)

        if self.optimizer_type == "adam":
            bc1 = 1.0 - self.beta1 ** self.step
            bc2 = 1.0 - self.beta2 ** self.step
            alpha_t = self.learning_rate * (np.sqrt(bc2) / bc1)
            eps_t = self.eps * np.sqrt(bc2)
            one_minus_b1 = 1.0 - self.beta1
            one_minus_b2 = 1.0 - self.beta2

            for i in range(len(self.weights)):
                gw = self.xp.clip(grad_w[i], -1.0, 1.0)
                gb = self.xp.clip(grad_b[i], -1.0, 1.0)
                self.m_w[i] = self.beta1 * self.m_w[i] + one_minus_b1 * gw
                self.v_w[i] = self.beta2 * self.v_w[i] + one_minus_b2 * (gw * gw)
                self.m_b[i] = self.beta1 * self.m_b[i] + one_minus_b1 * gb
                self.v_b[i] = self.beta2 * self.v_b[i] + one_minus_b2 * (gb * gb)

                self.weights[i] -= alpha_t * self.m_w[i] / (self.xp.sqrt(self.v_w[i]) + eps_t)
                self.biases[i] -= alpha_t * self.m_b[i] / (self.xp.sqrt(self.v_b[i]) + eps_t)
        else:
            for i in range(len(self.weights)):
                gw = self.xp.clip(grad_w[i], -1.0, 1.0)
                gb = self.xp.clip(grad_b[i], -1.0, 1.0)
                self.v_w[i] = self.momentum * self.v_w[i] + gw
                self.v_b[i] = self.momentum * self.v_b[i] + gb
                self.weights[i] -= self.learning_rate * self.v_w[i]
                self.biases[i] -= self.learning_rate * self.v_b[i]

        return loss

    def save_weights(self, filepath: Path) -> Path:
        filepath = Path(filepath).resolve()
        def to_cpu(arr):
            if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
            return np.asarray(arr)

        data_dict = {f"w_{idx}": to_cpu(w) for idx, w in enumerate(self.weights)}
        data_dict.update({f"b_{idx}": to_cpu(b) for idx, b in enumerate(self.biases)})
        data_dict["input_size"] = np.array(self.input_size)
        data_dict["hidden_layers"] = np.array(self.hidden_layers)
        data_dict["hidden_size"] = np.array(self.hidden_size)
        data_dict["output_size"] = np.array(self.output_size)
        data_dict["activation"] = np.array("leaky_relu")
        data_dict["leaky_relu_alpha"] = np.array(self.alpha)
        data_dict["use_residuals"] = np.array(True)
        data_dict["residual_scale"] = np.array(self.gamma)
        data_dict["loss_function"] = np.array("bce")
        data_dict["author"] = np.array("Petar Belchovski (P374R)")
        np.savez_compressed(filepath, **data_dict)
        return filepath

    save = save_weights


def generate_mock_genomic_data(num_samples: int = 200, sequence_length: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    bases = ["A", "C", "G", "T"]
    for i in range(num_samples):
        is_path = 1 if i % 2 == 1 else 0
        p = [0.10, 0.40, 0.40, 0.10] if is_path else [0.35, 0.15, 0.15, 0.35]
        gene = rng.choice(["TP53", "KRAS", "EGFR", "PIK3CA"]) if is_path else rng.choice(["ATM", "BRCA1", "BRCA2"])
        rows.append({
            "sequence": "".join(rng.choice(bases, size=sequence_length, p=p)),
            "is_cancerous": is_path,
            "GeneSymbol": gene,
            "Chromosome": str(rng.randint(1, 23)),
            "Start": rng.randint(100000, 200000000),
            "ReferenceAlleleVCF": "C" if is_path else "A",
            "AlternateAlleleVCF": "T" if is_path else "G",
            "Assembly": "GRCh38",
        })
    return pd.DataFrame(rows)


def train_network(
    data_path: Path = None, epochs: int = 2500, batch_size: int = 256,
    learning_rate: float = 0.001, optimizer: str = "adam", val_split: float = 0.20,
    output_weights_path: Path = None, device: str = "auto",
) -> tuple:
    if output_weights_path is None:
        output_weights_path = SCRIPT_DIR / "rotn_weights.npz"
    output_weights_path = Path(output_weights_path).resolve()

    print("\n+========================================================================+", flush=True)
    print("|  RESPONSIVE ONCOLOGY TEST NETWORK — DEEP RESIDUAL TRAINING ENGINE v2.2 |", flush=True)
    print("|  Author: Petar Belchovski (P374R)                                      |", flush=True)
    print("+========================================================================+\n", flush=True)

    device_lower = str(device).lower()
    if (device_lower == "auto" or device_lower == "gpu") and CUPY_AVAILABLE:
        resolved_device = "gpu"
    else:
        resolved_device = "cpu"

    if data_path and Path(data_path).is_file():
        resolved_data = Path(data_path).resolve()
        df = pd.read_csv(resolved_data)
    else:
        clinvar_csv = SCRIPT_DIR / "real_genomic_data.csv"
        mock_csv = SCRIPT_DIR / "mock_genomic_data.csv"
        if clinvar_csv.is_file():
            resolved_data = clinvar_csv
            df = pd.read_csv(resolved_data)
        elif mock_csv.is_file():
            resolved_data = mock_csv
            df = pd.read_csv(resolved_data)
        else:
            resolved_data = mock_csv
            df = generate_mock_genomic_data()
            df.to_csv(resolved_data, index=False)

    print(f"  [1/4] Loaded {len(df):,} records from: {resolved_data.name}", flush=True)
    t0_enc = time.time()
    X = encode_dataset_to_400dim(df)
    y = df["is_cancerous"].values.astype(np.float64).reshape(-1, 1)
    print(f"  [2/4] Encoded feature matrix {X.shape} in {time.time() - t0_enc:.3f}s", flush=True)

    indices = np.arange(len(df))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    split_boundary = int((1.0 - val_split) * len(df))
    X_train, y_train = X[indices[:split_boundary]], y[indices[:split_boundary]]
    X_val, y_val = X[indices[split_boundary:]], y[indices[split_boundary:]]
    print(f"        Train: {len(X_train):,} samples | Val: {len(X_val):,} samples", flush=True)

    print("  [3/4] Initializing 100-Layer Residual Network (437,569 parameters)...", flush=True)
    net = DeepGenomicNetwork(
        input_size=400, hidden_layers=100, hidden_size=64, output_size=1,
        learning_rate=learning_rate, optimizer=optimizer, device=resolved_device,
    )
    dev_str = "GPU (CUDA / CuPy)" if resolved_device == "gpu" else "CPU (Multi-threaded OpenBLAS / NumPy)"
    print(f"        Device: {dev_str} | Opt: {optimizer.upper()} | Batch: {batch_size}", flush=True)
    print(f"        Activation: LeakyReLU (alpha=0.01) | Residual Scale: {net.gamma:.4f} | Loss: BCE", flush=True)
    print("  " + "-" * 72, flush=True)

    # Adaptive reporting interval: responsive logging with live updates
    if epochs <= 25:
        report_interval = 1
    elif epochs <= 100:
        report_interval = 5
    elif epochs <= 500:
        report_interval = 10
    elif epochs <= 2500:
        report_interval = 25
    else:
        report_interval = 50

    t_start = time.time()
    n_train = len(X_train)
    batches_per_epoch = max(n_train // batch_size, 1)
    initial_train_bce, final_train_bce, final_val_bce = None, None, None
    best_val_bce, best_weights, best_biases = None, None, None

    train_eval_limit = min(n_train, 2000)

    if resolved_device == "gpu":
        X_train_dev = cp.asarray(X_train)
        y_train_dev = cp.asarray(y_train)
        X_val_dev = cp.asarray(X_val)
        y_val_dev = cp.asarray(y_val)
    else:
        X_train_dev = X_train
        y_train_dev = y_train
        X_val_dev = X_val
        y_val_dev = y_val

    try:
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n_train)
            if resolved_device == "gpu":
                perm_idx = cp.asarray(perm)
                X_shuff, y_shuff = X_train_dev[perm_idx], y_train_dev[perm_idx]
            else:
                X_shuff, y_shuff = X_train_dev[perm], y_train_dev[perm]

            for b_idx in range(batches_per_epoch):
                s_idx, e_idx = b_idx * batch_size, min((b_idx + 1) * batch_size, n_train)
                xb, yb = X_shuff[s_idx:e_idx], y_shuff[s_idx:e_idx]
                acts, zs = net.forward(xb)
                net.backpropagate(xb, yb, acts, zs)

            if epoch == 1 or epoch % report_interval == 0 or epoch == epochs:
                # Fast evaluation on subset for training + full validation set
                p_train = net.forward(X_train_dev[:train_eval_limit])[0][-1]
                train_bce = binary_cross_entropy_loss(p_train, y_train_dev[:train_eval_limit], xp=net.xp)
                train_acc = float(net.xp.mean((p_train >= 0.5) == y_train_dev[:train_eval_limit]))

                p_val = net.forward(X_val_dev)[0][-1]
                val_bce = binary_cross_entropy_loss(p_val, y_val_dev, xp=net.xp)
                val_acc = float(net.xp.mean((p_val >= 0.5) == y_val_dev))

                if initial_train_bce is None:
                    initial_train_bce = train_bce
                final_train_bce, final_val_bce = train_bce, val_bce

                # Save best checkpoint weights
                if best_val_bce is None or val_bce < best_val_bce:
                    best_val_bce = val_bce
                    best_weights = [w.copy() for w in net.weights]
                    best_biases = [b.copy() for b in net.biases]

                elapsed = max(time.time() - t_start, 0.001)
                sec_per_ep = elapsed / epoch
                rem_sec = max(int((epochs - epoch) * sec_per_ep), 0)
                rem_str = f"{rem_sec // 60}m {rem_sec % 60:02d}s" if rem_sec >= 60 else f"{rem_sec}s"
                print(f"  Epoch {epoch:4d}/{epochs} [{sec_per_ep:4.2f}s/ep, ETA: {rem_str:>6s}] | Train BCE: {train_bce:.4f} Acc: {train_acc*100:5.1f}% | Val BCE: {val_bce:.4f} Acc: {val_acc*100:5.1f}%", flush=True)

    except KeyboardInterrupt:
        print("\n\n  [!] Training paused by user (Ctrl+C). Saving current checkpoint...", flush=True)

    elapsed = max(time.time() - t_start, 0.001)
    print("  " + "-" * 72, flush=True)
    print(f"  Training finished in {elapsed:.1f}s ({elapsed / max(epoch, 1):.2f}s per epoch).", flush=True)
    if initial_train_bce and final_train_bce:
        reduction = ((initial_train_bce - final_train_bce) / initial_train_bce) * 100.0
        print(f"  Initial Train BCE: {initial_train_bce:.4f} -> Final Train BCE: {final_train_bce:.4f} (Reduction: {reduction:.1f}%) | Best Val BCE: {best_val_bce:.4f}\n", flush=True)

    # Restore best checkpoint if available
    if best_weights is not None:
        net.weights = best_weights
        net.biases = best_biases

    print(f"  [4/4] Serializing weights to: {output_weights_path.name}", flush=True)
    saved = net.save_weights(output_weights_path)
    print(f"        Saved {saved.stat().st_size / (1024 * 1024):.2f} MB archive.\n", flush=True)
    return net, saved


def main():
    parser = argparse.ArgumentParser(description="ROTN — Deep Residual Training Engine")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs (default: 2500).")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size (default: 256).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "momentum"], help="Optimizer choice (default: adam).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Compute device: auto, gpu, or cpu (default: auto).")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation split (default: 0.20).")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset.")
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "rotn_weights.npz"), help="Output .npz path.")
    args = parser.parse_args()

    train_network(
        data_path=args.data, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, optimizer=args.optimizer, val_split=args.val_split,
        output_weights_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()

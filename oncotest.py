"""
Responsive Oncology Test Network — Diagnostic Terminal Suite v2.2
===================================================================
Author: Petar Belchovski (P374R)
Copyright (c) 2026 Petar Belchovski (P374R). All Rights Reserved.
License: ROTN Non-Commercial Attribution & Experimental License (see LICENSE.txt)

A decoupled, zero-framework command-line interface for loading a pre-trained
100-layer deep residual genomic classifier and performing real-time mutation
pathogenicity predictions from raw DNA sequences or variant notation.
"""

import argparse
import ctypes
import io
import os
from pathlib import Path
import re
import sys
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def init_virtual_terminal() -> None:
    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            pass


init_virtual_terminal()


def clear_console() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


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


def encode_sequence_string(seq: str, max_len: int = 100) -> np.ndarray:
    vec = np.zeros((1, max_len * 4), dtype=np.float64)
    clean = seq.upper()
    for i in range(min(len(clean), max_len)):
        c = ord(clean[i]) if ord(clean[i]) < 256 else 0
        vec[0, i * 4 : i * 4 + 4] = NUCLEOTIDE_LUT[c]
    return vec


def parse_variant_string(s: str) -> tuple:
    s = s.strip()
    gene, chrom, pos, ref, alt = "", "", 0, "", ""
    m_chr = re.match(r"^(?:chr)?([0-9]{1,2}|[XYxy])[:_]([0-9]+)(?:[:_])?", s, re.IGNORECASE)
    if m_chr:
        chrom, pos, s = m_chr.group(1).upper(), int(m_chr.group(2)), s[m_chr.end():]
    else:
        m_gene = re.match(r"^([A-Za-z0-9_-]+):", s)
        if m_gene:
            cand = m_gene.group(1).upper()
            if cand.startswith("CHR"):
                chrom = cand.replace("CHR", "")
            elif cand in [str(i) for i in range(1, 23)] + ["X", "Y"]:
                chrom = cand
            else:
                gene = cand
            s = s[m_gene.end():]
            m_sub_chr = re.match(r"^(?:chr)?([0-9]{1,2}|[XYxy])[:_]([0-9]+)(?:[:_])?", s, re.IGNORECASE)
            if m_sub_chr:
                chrom, pos, s = m_sub_chr.group(1).upper(), int(m_sub_chr.group(2)), s[m_sub_chr.end():]
            elif m_cpos := re.search(r"c\.([0-9]+)", s):
                pos, s = int(m_cpos.group(1)), s[m_cpos.end():]

    if m_pos_allele := re.search(r"(?:([0-9]+))?([ACGT]+)(?:>|/|:)([ACGT]+)", s, re.IGNORECASE):
        if m_pos_allele.group(1) and pos == 0:
            pos = int(m_pos_allele.group(1))
        ref, alt = m_pos_allele.group(2).upper(), m_pos_allele.group(3).upper()
    return gene, chrom, pos, ref, alt


def sanitize_sequence_input(raw_text: str) -> tuple:
    lines = raw_text.strip().splitlines()
    had_fasta = False
    filtered = []
    for line in lines:
        l_s = line.strip()
        if l_s.startswith(">"):
            had_fasta = True
            continue
        filtered.append(l_s)
    merged = "".join(filtered)
    cleaned = "".join([c.upper() for c in merged if c.isalpha()])
    note = " (FASTA header removed)" if had_fasta else ""
    return cleaned, note


def parse_and_encode_input(raw_input: str) -> tuple:
    raw_clean = raw_input.strip()
    if (">" in raw_clean or "/" in raw_clean or ":" in raw_clean) and not raw_clean.startswith(">"):
        gene, chrom, pos, ref, alt = parse_variant_string(raw_clean)
        if ref and alt:
            vec = np.zeros((1, 400), dtype=np.float64)
            vec[0, 0:100] = encode_sequence_string(ref, max_len=25)[0]
            vec[0, 100:200] = encode_sequence_string(alt, max_len=25)[0]

            if len(ref) == 1 and len(alt) == 1 and ref in BASE_PAIR_INDICES and alt in BASE_PAIR_INDICES:
                vec[0, 200 + BASE_PAIR_INDICES[ref] * 4 + BASE_PAIR_INDICES[alt]] = 1.0
                if (ref, alt) in [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")]:
                    vec[0, 216] = 1.0
                else:
                    vec[0, 217] = 1.0

            if len(alt) > len(ref):
                vec[0, 218] = 1.0
            elif len(ref) > len(alt):
                vec[0, 219] = 1.0
            len_delta = len(alt) - len(ref)
            vec[0, 220] = np.tanh(len_delta / 10.0)
            vec[0, 221] = min(abs(len_delta), 50) / 50.0
            vec[0, 222] = 1.0 if "CG" in ref else 0.0
            vec[0, 223] = 1.0 if (ref, alt) in [("C", "T"), ("G", "A")] else 0.0
            vec[0, 224] = (ref.count("G") + ref.count("C")) / max(len(ref), 1)
            vec[0, 225] = (alt.count("G") + alt.count("C")) / max(len(alt), 1)

            if chrom in CHROM_TO_INDEX:
                vec[0, 226 + CHROM_TO_INDEX[chrom]] = 1.0
            vec[0, 250] = min(pos, 250000000) / 250000000.0
            if gene in GENE_TO_INDEX:
                vec[0, 251 + GENE_TO_INDEX[gene]] = 1.0
            elif gene:
                h = hash(gene)
                for k in range(12):
                    vec[0, 271 + k] = 1.0 if ((h >> (k * 2)) & 1) else 0.0

            d_gene = gene if gene else ("Chr" + chrom if chrom else "")
            d_pos = f":{pos}" if pos else ""
            d_str = f"{d_gene}{d_pos}:{ref}>{alt}" if d_gene else f"{ref}>{alt}"
            meta = {
                "type": "variant", "display": d_str, "ref": ref, "alt": alt,
                "gene": gene or ("Chr " + chrom if chrom else "N/A"), "chrom": chrom or "N/A", "pos": pos,
                "length": len(ref) + len(alt),
                "gc_content": ((ref.count("G") + ref.count("C") + alt.count("G") + alt.count("C")) / max(len(ref) + len(alt), 1)) * 100.0,
                "A": ref.count("A") + alt.count("A"), "C": ref.count("C") + alt.count("C"),
                "G": ref.count("G") + alt.count("G"), "T": ref.count("T") + alt.count("T"),
                "N": sum(1 for c in (ref + alt) if c not in "ACGT"),
            }
            return vec, meta

    clean_seq, _ = sanitize_sequence_input(raw_input)
    vec = encode_sequence_string(clean_seq, max_len=100)
    total = len(clean_seq)
    meta = {
        "type": "sequence", "display": clean_seq, "ref": clean_seq[:min(total, 25)], "alt": "",
        "gene": "N/A", "chrom": "N/A", "pos": 0, "length": total,
        "gc_content": ((clean_seq.count("G") + clean_seq.count("C")) / max(total, 1)) * 100.0,
        "A": clean_seq.count("A"), "C": clean_seq.count("C"),
        "G": clean_seq.count("G"), "T": clean_seq.count("T"),
        "N": sum(1 for c in clean_seq if c not in "ACGT"),
    }
    return vec, meta


def get_desktop_directories() -> list:
    candidates = []
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")
            val, _ = winreg.QueryValueEx(k, "Desktop")
            exp = Path(os.path.expandvars(val)).resolve()
            if exp.is_dir():
                candidates.append(exp)
        except Exception:
            pass
    home = Path.home()
    candidates.extend([home / "OneDrive" / "Desktop", home / "Desktop"])
    up = os.environ.get("USERPROFILE")
    if up:
        candidates.extend([Path(up) / "OneDrive" / "Desktop", Path(up) / "Desktop"])
    existing = []
    for p in candidates:
        if p.is_dir() and p not in existing:
            existing.append(p)
    return existing


def locate_resource(filename: str = "rotn_weights.npz") -> tuple:
    target = Path(filename)
    if target.is_file():
        return target.resolve(), [target.parent.resolve()]
    search_roots = [Path.cwd(), SCRIPT_DIR] + get_desktop_directories() + [Path.home() / "Downloads", Path.home() / "Documents"]
    deduped = []
    for r in search_roots:
        if r.is_dir() and r not in deduped:
            deduped.append(r)
    searched = []
    for root in deduped:
        searched.append(str(root))
        if (root / filename).is_file():
            return (root / filename).resolve(), searched
        for sub in ["models", "weights", "Responsive-Oncology-Test-Network"]:
            if (root / sub / filename).is_file():
                return (root / sub / filename).resolve(), searched
    return None, searched


class ReconstructedNetwork:
    def __init__(self, weights_path: Path):
        resolved, searched = locate_resource(str(weights_path))
        if not resolved:
            print(f"\n  [ERROR] Model weights archive not found! Target: '{weights_path}'")
            print("  Searched: " + ", ".join(searched[:3]))
            print("  Execute 'python train.py' to generate weights.\n")
            sys.exit(1)
        self.weights_path = resolved
        archive = np.load(self.weights_path, allow_pickle=True)
        self.weights, self.biases = [], []
        idx = 0
        while f"w_{idx}" in archive:
            self.weights.append(archive[f"w_{idx}"])
            self.biases.append(archive[f"b_{idx}"])
            idx += 1
        self.hidden_layers = len(self.weights) - 1
        self.use_residuals = bool(archive.get("use_residuals", True))
        self.gamma = float(archive.get("residual_scale", 1.0 / np.sqrt(self.hidden_layers)))
        self.alpha = float(archive.get("leaky_relu_alpha", 0.01))
        self._total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    @property
    def total_parameters(self) -> int:
        return self._total_params

    def forward(self, X: np.ndarray) -> np.ndarray:
        z0 = X @ self.weights[0] + self.biases[0]
        cur = np.where(z0 > 0, z0, self.alpha * z0)
        for l in range(1, self.hidden_layers):
            z = cur @ self.weights[l] + self.biases[l]
            h = np.where(z > 0, z, self.alpha * z)
            cur = cur + self.gamma * h if self.use_residuals else h
        z_out = cur @ self.weights[-1] + self.biases[-1]
        return 1.0 / (1.0 + np.exp(-np.clip(z_out, -500.0, 500.0)))


def classify_risk(prob: float) -> tuple:
    if prob >= 0.90:
        return ("PATHOGENIC -- CRITICAL", "XXXXXXXXXXXXXXXXXXXXX CRITICAL", "Extremely high pathogenicity signal detected. Strong computational signature of deleterious genomic alteration. Immediate expert review recommended.")
    elif prob >= 0.75:
        return ("PATHOGENIC -- HIGH RISK", "XXXXXXXXXXXXXXXX----- HIGH", "Strong pathogenicity signal detected. Likely pathogenic profile. Further clinical investigation and confirmatory testing advised.")
    elif prob >= 0.50:
        return ("VARIANT OF UNCERTAIN SIGNIFICANCE (VUS)", "XXXXXXXXXXX---------- ELEVATED", "Moderate pathogenicity signal detected. Variant falls in uncertain significance zone. Additional functional assays advised.")
    elif prob >= 0.25:
        return ("LIKELY BENIGN", "XXXXX---------------- LOW", "Weak pathogenicity signal detected. Mutation profile suggests predominantly benign character. Routine monitoring sufficient.")
    elif prob >= 0.10:
        return ("BENIGN -- LOW RISK", "XX------------------- MINIMAL", "Very weak pathogenicity signal. Computationally benign with minimal risk indicators. Standard monitoring recommended.")
    return ("BENIGN -- NOMINAL", "--------------------- NOMINAL", "Nominal baseline genomic profile. No significant pathogenic indicators detected across 100 deep residual representation layers.")


def print_diagnostic_report(meta: dict, prob: float, query_number: int) -> None:
    cat, bar, desc = classify_risk(prob)
    disp = meta.get("display", "N/A")
    if len(disp) > 48:
        disp = disp[:22] + "..." + disp[-22:]
    tag = "[!! HAZARD ALERT !!]" if prob >= 0.75 else ("[! CAUTION !]" if prob >= 0.50 else ("[MONITOR]" if prob >= 0.25 else "[CLEAR]"))

    print("\n+" + "-" * 70 + "+")
    print(f"|  DIAGNOSTIC REPORT -- Query #{query_number:<40d}  |")
    print("|" + "-" * 70 + "|")
    print(f"|  Hazard Status   : {tag:<47s}  |")
    print("|" + "-" * 70 + "|")
    print(f"|  Target Input    : {disp:<47s}  |")
    if meta.get("gene") != "N/A":
        g_info = f"{meta['gene']} (Chr {meta.get('chrom', '?')}:{meta.get('pos', 0)})"
        print(f"|  Gene Context    : {g_info:<47s}  |")
    print(f"|  Sequence Length : {meta['length']:<47d}  |")
    print(f"|  Nucleotide A    : {meta['A']:<47d}  |")
    print(f"|  Nucleotide C    : {meta['C']:<47d}  |")
    print(f"|  Nucleotide G    : {meta['G']:<47d}  |")
    print(f"|  Nucleotide T    : {meta['T']:<47d}  |")
    print(f"|  Ambiguous (N)   : {meta['N']:<47d}  |")
    print(f"|  GC Content      : {meta['gc_content']:.2f}%{'':<41s}  |")
    print("|" + "-" * 70 + "|")
    print(f"|  Raw Probability : {prob:.8f}{'':<37s}  |")
    print(f"|  Percentage Risk : {prob * 100:.4f}%{'':<39s}  |")
    print(f"|  Risk Level      : {bar:<47s}  |")
    print("|" + "-" * 70 + "|")
    print(f"|  CLASSIFICATION  : {cat:<47s}  |")
    print("|" + "-" * 70 + "|")
    print(f"|{'':<70s}|")
    line = ""
    for w in desc.split():
        if len(line) + len(w) + 1 <= 66:
            line = line + " " + w if line else w
        else:
            print(f"|  {line:<66s}  |")
            line = w
    if line:
        print(f"|  {line:<66s}  |")
    print(f"|{'':<70s}|")
    print("+" + "-" * 70 + "+\n")


BENCHMARK_CASES = [
    ("TP53 Hotspot Oncogenic Variant (c.742C>T, p.R248W)", "TP53:c.742C>T"),
    ("KRAS Codon 12 Oncogenic Driver (c.35G>A, p.G12D)", "KRAS:c.35G>A"),
    ("ATM Intronic Benign Polymorphism Reference", "ATM:chr11:108223000:A:G"),
    ("BRCA1 Benign Intronic Polymorphic Region", "AATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"),
]


def run_benchmark_demo(net: ReconstructedNetwork) -> None:
    print("\n+" + "=" * 70 + "+")
    print("|  RUNNING CLINICAL BENCHMARK DEMONSTRATION                          |")
    print("+" + "=" * 70 + "+")
    for idx, (label, item) in enumerate(BENCHMARK_CASES, start=1):
        print(f"\n  [BENCHMARK CASE #{idx}] {label}\n  Input: '{item}'")
        vec, meta = parse_and_encode_input(item)
        print_diagnostic_report(meta, float(net.forward(vec)[0, 0]), idx)


def run_interactive_session(net: ReconstructedNetwork) -> None:
    clear_console()
    print("+" + "=" * 70 + "+")
    print("|  RESPONSIVE ONCOLOGY TEST NETWORK                                  |")
    print("|  Diagnostic Terminal Suite v2.2                                    |")
    print("|  Author: Petar Belchovski (P374R)                                  |")
    print("|  Zero-Framework 100-Layer Deep Residual Classifier                 |")
    print("+" + "=" * 70 + "+\n")
    print(f"  Model loaded: {net.weights_path.name}")
    print(f"  Parameters  : {net.total_parameters:,} across 100 residual layers")
    print("  Commands    : 'demo' (benchmarks), 'cls' (clear), 'help', 'quit'\n")

    query_count = 0
    while True:
        try:
            user_input = input("ROTN> Enter DNA sequence or variant notation: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Terminating diagnostic session. Goodbye!")
            break
        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ["quit", "exit", "q"]:
            print("  Terminating diagnostic session. Goodbye!")
            break
        elif cmd in ["cls", "clear"]:
            clear_console()
            continue
        elif cmd in ["help", "?"]:
            print("\n  Usage: Enter DNA sequence (e.g. CGCCGCCG...) or variant (e.g. TP53:c.742C>T, C>T).")
            print("         Type 'demo' for clinical benchmarks, 'cls' to clear screen, 'quit' to exit.\n")
            continue
        elif cmd in ["demo", "test"]:
            run_benchmark_demo(net)
            continue

        query_count += 1
        vec, meta = parse_and_encode_input(user_input)
        print_diagnostic_report(meta, float(net.forward(vec)[0, 0]), query_count)


def main():
    parser = argparse.ArgumentParser(description="Responsive Oncology Test Network — Diagnostic Terminal Suite")
    parser.add_argument("sequence_or_variant", nargs="?", default=None, help="Target DNA sequence or variant notation (or '-' for piped STDIN).")
    parser.add_argument("--demo", action="store_true", help="Run pre-loaded clinical benchmark demonstrations.")
    parser.add_argument("--weights", type=str, default="rotn_weights.npz", help="Path to trained weights .npz archive.")
    args = parser.parse_args()

    net = ReconstructedNetwork(Path(args.weights))
    if args.demo:
        clear_console()
        run_benchmark_demo(net)
    elif args.sequence_or_variant == "-":
        if piped := sys.stdin.read().strip():
            vec, meta = parse_and_encode_input(piped)
            print_diagnostic_report(meta, float(net.forward(vec)[0, 0]), 1)
    elif args.sequence_or_variant:
        vec, meta = parse_and_encode_input(args.sequence_or_variant)
        print_diagnostic_report(meta, float(net.forward(vec)[0, 0]), 1)
    else:
        run_interactive_session(net)


if __name__ == "__main__":
    main()

# The Price of Parsability

**DASE7506 Advanced Machine Learning Methods — Final Project (HKU)**

This repository contains everything needed to reproduce every number, table and figure in the report *The Price of Parsability: How Output-Format Constraints Change Accuracy and Cost in Mathematical Reasoning*.

> **Repository:** https://github.com/huangxv133/dase7506-format-tax
> **Submitted PDF:** `DASE7506_3036705931_Huang_Xu.pdf` (also at `paper/paper.pdf`)
>
> ```bash
> git clone https://github.com/huangxv133/dase7506-format-tax.git
> cd dase7506-format-tax
> ```
>
> To check the reported numbers without a GPU: unpack `data/all_runs.tgz`, run the two commands in §5.2, and diff against `out/`. Grading is deterministic, so the tables must match exactly.

The study measures how output-format constraints (an unconstrained ending, a `Final answer:` line, a `\boxed{}` box) interact with verbosity instructions (full vs. brief reasoning) in zero-shot mathematical reasoning. Each response is graded twice — by a *strict* extractor that honours only the requested format and a *lenient* extractor that accepts any standard format — which separates a **parsing tax** (accuracy lost because the evaluator could not read a correct answer) from a **capability tax** (a real change in the model's reasoning).

---

## 1. Results at a glance

3 models × 7 conditions × 200 GSM8K problems = **4,200 greedy generations**.

| Finding | Number |
|---|---|
| `\boxed{}` format, parsing tax | ≤ 0.5 pp at all three model sizes, compliance 96–99% |
| `Final answer:` line, parsing tax | 41.0 / 18.0 / 12.5 pp at 0.5B / 1.5B / 7B |
| Capability tax of the answer line | not significant at any size (+0.5 / +5.0 / −3.0 pp) |
| Brevity instruction, 7B | −11.5 pp for 203 fewer tokens (5.7 pp per 100 tokens saved) |
| Brevity + `\boxed{}`, 7B | −23.5 pp, super-additive; 36.5% of responses < 20 tokens |

The headline claim: the cost of a format constraint is paid almost entirely by the evaluator, not by the model, and it depends on *which* format is requested rather than on whether one is requested at all.

---

## 2. Repository layout

```
code/run.py          generation: 7 prompt conditions × N problems × one model → JSONL
code/reextract.py    frozen extractors; re-grades stored raw text (no GPU, seconds)
code/analyze.py      paired statistics, three tables, three figures
data/all_runs.tgz    all 21 JSONL logs (raw model output for every item)
data/                three readable sample logs (C1, C2, C6 of two models)
figs/                fig1_heatmap.png, fig2_pareto.png, fig3_tax.png (300 dpi)
out/metrics.csv      per-(model, condition) metrics
out/tables.md        Table I / II / III and the tax decomposition
paper/paper.md       report source (Markdown)
paper/appendix.md    appendix source (verbatim prompts, extractor rules)
paper/build_pdf.py   Markdown → IEEE-style PDF (Chrome headless print)
paper/paper.pdf      the submitted report (DASE7506_3036705931_Huang_Xu.pdf)
```

---

## 3. Environment

Tested on Ubuntu 22.04 with 1× NVIDIA A800-80GB (any GPU with ≥ 16 GB works; the 7B model in bfloat16 needs ~15 GB of weights).

| Package | Version used |
|---|---|
| Python | 3.10 |
| torch | 2.8.0 + cu128 (must match the driver; cu130 fails on driver 570) |
| transformers | 4.46 |
| datasets | 3.x |
| accelerate, scipy, numpy, matplotlib | current releases |
| modelscope | for model download (see below) |

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate scipy numpy matplotlib modelscope
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `pip install torch` resolves to a CUDA build newer than your driver, pin the build explicitly — a CUDA-13 wheel on a 12.8 driver installs cleanly but `torch.cuda.is_available()` returns `False`.

---

## 4. Data and models

**GSM8K** (test split, 1,319 problems) is streamed from the Hugging Face hub. The first 200 problems after a fixed shuffle with seed `7506` are used, in the same order for every condition and every model.

```bash
export HF_ENDPOINT=https://hf-mirror.com   # only if huggingface.co is unreachable
python -c "from datasets import load_dataset; d=load_dataset('openai/gsm8k','main',split='test'); print(len(d))"
```

**Models** — Qwen2.5-Instruct at three sizes, downloaded from ModelScope (much faster than the HF mirror from mainland China):

```bash
python - <<'PY'
from modelscope import snapshot_download
for mid in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
    print(snapshot_download(mid, cache_dir="./models"))
PY
```

---

## 5. Reproducing the results

### 5.1 Regenerate the grid (GPU, ~15 min on three A800s)

One process per model, one GPU each:

```bash
B=./models
M5=$(ls -d  $B/Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ | head -1)
M15=$(ls -d $B/Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/ | head -1)
M7=$(ls -d  $B/Qwen--Qwen2.5-7B-Instruct/snapshots/*/  | head -1)

export HF_ENDPOINT=https://hf-mirror.com
python code/run.py --model "$M5"  --tag qwen0.5b --n 200 --gpu 0 --batch 32 --out runs/main &
python code/run.py --model "$M15" --tag qwen1.5b --n 200 --gpu 1 --batch 32 --out runs/main &
python code/run.py --model "$M7"  --tag qwen7b  --n 200 --gpu 2 --batch 24 --out runs/main &
wait
```

`run.py` writes one JSONL per (model, condition) with the raw response, token count, truncation flag and both extraction results.

### 5.2 Re-grade from the stored logs (CPU, seconds — no model needed)

The submitted numbers come from this step, not from the generation step, so the whole analysis can be checked without a GPU:

```bash
mkdir -p data/raw && tar xzf data/all_runs.tgz -C data/raw
python code/reextract.py --run data/raw
python code/analyze.py   --run data/raw --out out/main
```

Expected output: `out/main/metrics.csv`, `out/main/tables.md`, and the three figures. The values must match `out/` in this repository exactly; grading is deterministic given the stored text.

### 5.3 Rebuild the PDF (no LaTeX required)

```bash
pip install pypdfium2    # only needed for the page-count report
python paper/build_pdf.py \
  --author "Huang Xu · 3036705931" \
  --repo   "https://github.com/huangxv133/dase7506-format-tax" \
  --out    paper/paper.pdf
```

The script renders `paper/paper.md` + `paper/appendix.md` to IEEE-style HTML and prints it with headless Chrome; it needs Chrome installed but no LaTeX toolchain.

---

## 6. Condition map

All conditions share the system message *"You are a helpful assistant."*; only the final instruction sentence differs.

| | free ending | `Final answer:` line | `\boxed{}` |
|---|---|---|---|
| **full steps** | C1 (baseline) | C2 | C3 |
| **brief steps** | C4 | C5 | C6 |

C0 = no reasoning at all (answer only), used as a sensitivity anchor. Verbatim prompts are in Appendix A of the report and in `paper/appendix.md`.

---

## 7. Reproducibility notes and known limits

- **Extractors are frozen.** `reextract.py` is the version used for every number in the report. The lenient extractor tries `\boxed{}`, then an answer line, then a trailing number (the latter only if it falls in the last 20% of the response; responses ≤ 60 characters are exempt). Unconstrained conditions (C0, C1, C4) have strict score = lenient score by definition, so no parsing tax can arise from having no format instruction.
- **Re-grading never re-runs the model.** Any change to the extraction rules can be re-applied to the stored logs, which is why the raw text is released in full.
- **200 problems.** Paired 95% intervals (MOVER–Newcombe) are ±5–8 pp, enough for the large effects reported; the 0.5B brevity difference is not resolvable at this sample size.
- **Absolute accuracies are inflated** by GSM8K's likely presence in pretraining data; the paper's claims are about paired contrasts, not absolute levels.

---

## 8. License

Code released under the MIT License. The report text and figures are the author's own work, submitted for DASE7506 at The University of Hong Kong.

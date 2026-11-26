"""分析：双抽取器指标、能力税/解析税、汇率、配对 CI、翻转表、分层、作图。

输出：metrics.csv、tables.md、fig1_heatmap.png、fig2_pareto.png、fig3_tax.png
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
LABEL = {"C0": "C0 无推理", "C1": "C1 完整/自由", "C2": "C2 完整/模板行",
         "C3": "C3 完整/boxed", "C4": "C4 简短/自由", "C5": "C5 简短/模板行",
         "C6": "C6 简短/boxed"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe_paired(n11, n10, n01, n00):
    """MOVER-Newcombe 配对比例差的 95% CI。X=C1 型（基准），Y=对照。"""
    n = n11 + n10 + n01 + n00
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    pX = (n11 + n10) / n
    pY = (n11 + n01) / n
    d = pX - pY
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    lo = d - math.sqrt(max(0.0, (pX - l1) ** 2 + (u2 - pY) ** 2))
    hi = d + math.sqrt(max(0.0, (u1 - pX) ** 2 + (pY - l2) ** 2))
    return (d, lo, hi)


def load(run_dir):
    rows = []
    for fp in sorted(glob.glob(str(Path(run_dir) / "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.run)
    print(f"[load] {len(rows)} 条")

    by = defaultdict(dict)          # (tag,cond) -> uid -> row
    for r in rows:
        by[(r["model"], r["cond"])][r["uid"]] = r
    tags = sorted({r["model"] for r in rows})
    print("[models]", tags)

    # ---------------- 逐条件指标
    met = {}
    for (tag, cond), d in by.items():
        n = len(d)
        al = sum(v["lenient_ok"] for v in d.values()) / n
        astr = sum(v["strict_ok"] for v in d.values()) / n
        anp = sum(v["lenient_nopos_ok"] for v in d.values()) / n
        comp = sum(1 for v in d.values() if v["strict_pred"] is not None) / n
        fail = sum(1 for v in d.values() if v["strict_pred"] is None
                   and v["lenient_pred"] is None) / n
        tk = sum(v["n_out"] for v in d.values()) / n
        tr = sum(1 for v in d.values() if v["truncated"]) / n
        met[(tag, cond)] = dict(n=n, acc_len=al, acc_str=astr, acc_nopos=anp,
                                comp=comp, fail=fail, tok=tk, trunc=tr,
                                rate=al / tk * 1000 if tk else float("nan"))

    # ---------------- 表
    lines = []
    lines.append("| 模型 | 条件 | 宽容准确率 | 严格准确率 | 合规率 | 双失败率 | 输出token | 每千token准确率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for tag in tags:
        for cond in CONDS:
            m = met.get((tag, cond))
            if not m:
                continue
            lines.append(f"| {tag} | {LABEL[cond]} | {m['acc_len']:.3f} | {m['acc_str']:.3f} | "
                         f"{m['comp']:.3f} | {m['fail']:.3f} | {m['tok']:.0f} | {m['rate']:.2f} |")

    # ---------------- 税与汇率（相对 C1）
    lines.append("\n| 模型 | 对照 | 能力税 Δcap (pp) | 95% CI | 解析税 Δparse (pp) | 翻转 对→错 / 错→对 | token 变化 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    tax = {}
    for tag in tags:
        base = by.get((tag, "C1"), {})
        for cond in [c for c in CONDS if c != "C1"]:
            cur = by.get((tag, cond), {})
            common = sorted(set(base) & set(cur))
            if not common:
                continue
            n11 = sum(1 for u in common if base[u]["lenient_ok"] and cur[u]["lenient_ok"])
            n10 = sum(1 for u in common if base[u]["lenient_ok"] and not cur[u]["lenient_ok"])
            n01 = sum(1 for u in common if not base[u]["lenient_ok"] and cur[u]["lenient_ok"])
            n00 = len(common) - n11 - n10 - n01
            d, lo, hi = newcombe_paired(n11, n10, n01, n00)
            cap = -d * 100                       # 能力税 = 基准 - 对照
            cap_lo, cap_hi = -hi * 100, -lo * 100  # 翻转符号后区间两端互换
            parse = (met[(tag, cond)]["acc_len"] - met[(tag, cond)]["acc_str"]) * 100
            dt = met[(tag, cond)]["tok"] - met[(tag, "C1")]["tok"]
            tax[(tag, cond)] = dict(cap=cap, lo=cap_lo, hi=cap_hi,
                                    parse=parse, dtok=dt,
                                    rate=(cap / (-dt / 100)) if dt < 0 else float("nan"))
            lines.append(f"| {tag} | {LABEL[cond]} | {cap:+.2f} | [{cap_lo:+.2f}, {cap_hi:+.2f}] | "
                         f"{parse:+.2f} | {n10} / {n01} | {dt:+.0f} |")

    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- CSV
    with open(out / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("model,cond,n,acc_len,acc_str,acc_nopos,comp,fail,tok,trunc,rate\n")
        for (tag, cond), m in met.items():
            f.write(f"{tag},{cond},{m['n']},{m['acc_len']:.4f},{m['acc_str']:.4f},"
                    f"{m['acc_nopos']:.4f},{m['comp']:.4f},{m['fail']:.4f},"
                    f"{m['tok']:.1f},{m['trunc']:.4f},{m['rate']:.3f}\n")

    # ---------------- 图1 热力图
    fmt = ["C1", "C2", "C3", "C4", "C5", "C6"]
    NICE = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "qwen7b": "Qwen2.5-7B"}
    fig, axes = plt.subplots(1, len(tags), figsize=(3.4 * len(tags), 2.3), sharey=True)
    if len(tags) == 1:
        axes = [axes]
    for k, (ax, tag) in enumerate(zip(axes, tags)):
        M = np.full((2, 3), np.nan)
        for i, d in enumerate(["C1", "C2", "C3"]):
            M[0, i] = met[(tag, d)]["acc_len"]
        for i, d in enumerate(["C4", "C5", "C6"]):
            M[1, i] = met[(tag, d)]["acc_len"]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{M[i,j]*100:.1f}", ha="center", va="center",
                        fontsize=8.5, color="white" if M[i, j] > 0.55 else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["free", "answer line", "boxed"], fontsize=7.5)
        ax.set_title(NICE.get(tag, tag), fontsize=9)
        ax.set_yticks(range(2))
        if k == 0:
            ax.set_yticklabels(["full\nsteps", "brief"], fontsize=8)
    fig.subplots_adjust(wspace=0.12)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("lenient accuracy", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out / "fig1_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图2 准确率-成本
    PAL = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, tag in enumerate(tags):
        xs = [met[(tag, c)]["tok"] for c in fmt if (tag, c) in met]
        ys = [met[(tag, c)]["acc_len"] * 100 for c in fmt if (tag, c) in met]
        ax.scatter(xs, ys, s=42, color=PAL[k % len(PAL)], label=NICE.get(tag, tag),
                   edgecolor="white", linewidth=0.6, zorder=3)
        for j, (x, y, c) in enumerate(zip(xs, ys, [c for c in fmt if (tag, c) in met])):
            dy = 4 if j % 2 == 0 else -10
            ax.annotate(c, (x, y), fontsize=6.5, color=PAL[k % len(PAL)],
                        xytext=(4, dy), textcoords="offset points")
    ax.set_xlabel("average output tokens per question", fontsize=8.5)
    ax.set_ylabel("lenient accuracy (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(50, 360)
    ax.set_ylim(40, 100)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.savefig(out / "fig2_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图3 能力税
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    conds = ["C2", "C3", "C4", "C5", "C6"]
    w = 0.8 / max(1, len(tags))
    for k, tag in enumerate(tags):
        vals = [tax.get((tag, c), {}).get("cap", 0) for c in conds]
        pos = np.arange(len(conds)) + (k - (len(tags) - 1) / 2) * w
        ax.bar(pos, vals, w * 0.92, label=NICE.get(tag, tag), color=PAL[k % len(PAL)])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("capability tax vs C1 (pp)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(out / "fig3_tax.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 分层
    strata = ["1–3 步", "4+ 步"]
    lines.append("\n| 模型 | 条件 | 1–3 步 宽容准确率 (n) | 4+ 步 宽容准确率 (n) |")
    lines.append("| --- | --- | --- | --- |")
    for tag in tags:
        for cond in ["C1", "C4", "C5"]:
            d = by.get((tag, cond), {})
            if not d:
                continue
            s1 = [v for v in d.values() if v["n_steps"] <= 3]
            s2 = [v for v in d.values() if v["n_steps"] > 3]
            f1 = (sum(v["lenient_ok"] for v in s1) / len(s1) * 100) if s1 else float("nan")
            f2 = (sum(v["lenient_ok"] for v in s2) / len(s2) * 100) if s2 else float("nan")
            lines.append(f"| {tag} | {LABEL[cond]} | {f1:.1f} ({len(s1)}) | {f2:.1f} ({len(s2)}) |")

    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("[done]", "\n".join(lines[:8]))


if __name__ == "__main__":
    main()

"""DASE7506 实验：格式约束的代价（Generation）

7 个条件 × N 题 × 3 个模型。贪心解码，max_new_tokens=512。
输出 JSONL，每条含原始输出、token 数、finish_reason、两套抽取器的结果。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SYSTEM = "You are a helpful assistant."

INSTR = {
    "C0": "Answer the question. Output only the final number on a single line, with no other text.",
    "C1": "Solve the problem. Show your calculation steps in full.",
    "C2": "Solve the problem. Show your calculation steps in full. End with one line in the form \"Final answer: <number>\".",
    "C3": "Solve the problem. Show your calculation steps in full. Put the final number inside \\boxed{}, for example \\boxed{42}.",
    "C4": "Solve the problem. Show only the essential calculation steps, as briefly as possible.",
    "C5": "Solve the problem. Show only the essential calculation steps, as briefly as possible. End with one line in the form \"Final answer: <number>\".",
    "C6": "Solve the problem. Show only the essential calculation steps, as briefly as possible. Put the final number inside \\boxed{}, for example \\boxed{42}.",
}
CONDS = list(INSTR)


# ------------------------------------------------------------------ 抽取器

_BOX = re.compile(r"\\boxed\s*\{")
_FA = re.compile(r"Final answer:\s*(.+)", re.I)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _match_braces(s, start):
    depth, i = 1, start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i]
        i += 1
    return None


def _to_num(s):
    if s is None:
        return None
    t = s.strip()
    t = t.strip("$").strip()
    t = t.replace(",", "").replace(" ", "")
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\%", "").rstrip("%").rstrip(".")
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?", t)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", t)
    if m:
        d = float(m.group(2))
        return float(m.group(1)) / d if d else None
    m = re.fullmatch(r"-?\d[\d,]*\.?\d*", t)
    if m:
        try:
            return float(t.replace(",", ""))
        except ValueError:
            return None
    return None


def _last_boxed(text):
    last = None
    for m in _BOX.finditer(text):
        inner = _match_braces(text, m.end())
        if inner is not None:
            last = inner
    return _to_num(last) if last is not None else None


def _last_answer_line(text):
    ms = _FA.findall(text)
    return _to_num(ms[-1]) if ms else None


def _tail_number(text):
    """全文最后一个数值，且须落在末尾 20% 字符区间内（否则视为歧义）。

    短输出（如只有 "14"）不做位置约束，否则会被误判为抽不到。
    """
    ms = list(NUM.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    if len(text) > 60 and m.start() < 0.8 * len(text):
        return None
    return _to_num(m.group(0))


def extract_strict(text, cond):
    """严格：只认该条件被要求的那一种格式；不符即记格式失败（None）。"""
    if cond in ("C2", "C5"):
        return _last_answer_line(text)
    if cond in ("C3", "C6"):
        return _last_boxed(text)
    return _tail_number(text)


def extract_lenient(text):
    """宽容：不惩罚"没按指定格式写"，依次尝试 boxed → answer 行 → 尾部数值。

    这是"一个不关心格式的评测者"会给出的判分，因此恒有 lenient >= strict。
    """
    for f in (_last_boxed, _last_answer_line, _tail_number):
        v = f(text)
        if v is not None:
            return v
    return None


def extract_lenient_nopos(text):
    """敏感性分析：不施加位置约束的尾部数值。"""
    ms = list(NUM.finditer(text))
    return _to_num(ms[-1].group(0)) if ms else None


def ok(pred, gold, tol=1e-4):
    if pred is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


# ------------------------------------------------------------------ 数据

def load_gsm8k_test(n, seed):
    import random

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    out = []
    for i in idx:
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip()
        steps = len(re.findall(r"<<[^>]+>>", r["answer"]))
        out.append({
            "uid": f"gsm8k/{i}",
            "qid": i,
            "problem": r["question"],
            "gold": float(gold.replace(",", "")),
            "n_steps": steps,
        })
    return out


# ------------------------------------------------------------------ 生成

def build_inputs(tok, items, cond):
    texts = []
    for it in items:
        user = it["problem"].strip() + "\n\n" + INSTR[cond]
        texts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7506)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    items = load_gsm8k_test(args.n, args.seed)
    print(f"[data] {len(items)} 题", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    _kw = dict(device_map={"": args.gpu}, trust_remote_code=True)
    try:  # transformers>=4.56 用 dtype，旧版用 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, **_kw).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, **_kw).eval()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    conds = args.conds.split(",")

    for cond in conds:
        path = outdir / f"{args.tag}_{cond}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            print(f"[skip] {path}", flush=True)
            continue
        texts = build_inputs(tok, items, cond)
        rows = []
        t0 = time.time()
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(f"cuda:{args.gpu}")
            with torch.inference_mode():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, it in enumerate(items[i:i + args.batch]):
                ids = new[j].tolist()
                while ids and ids[-1] == tok.pad_token_id:
                    ids.pop()
                text = tok.decode(ids, skip_special_tokens=True)
                trunc = len(ids) >= args.max_new
                sp = extract_strict(text, cond)
                lp = extract_lenient(text)
                lnp = extract_lenient_nopos(text)
                rows.append({
                    "model": args.tag, "cond": cond, "uid": it["uid"],
                    "qid": it["qid"], "n_steps": it["n_steps"], "gold": it["gold"],
                    "output": text, "n_out": len(ids),
                    "truncated": bool(trunc),
                    "strict_pred": sp, "lenient_pred": lp, "lenient_nopos_pred": lnp,
                    "strict_ok": ok(sp, it["gold"]),
                    "lenient_ok": ok(lp, it["gold"]),
                    "lenient_nopos_ok": ok(lnp, it["gold"]),
                })
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc_l = sum(r["lenient_ok"] for r in rows) / len(rows)
        acc_s = sum(r["strict_ok"] for r in rows) / len(rows)
        tok_avg = sum(r["n_out"] for r in rows) / len(rows)
        print(f"[{args.tag} {cond}] lenient={acc_l:.3f} strict={acc_s:.3f} "
              f"tok={tok_avg:.0f} trunc={sum(r['truncated'] for r in rows)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()

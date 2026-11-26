"""重算判分：只读已存盘的原始输出，不碰模型。

抽取器冻结版本 v1.0（正式结果以本文件为准）：
  strict  —— 只认该条件被要求的那一种格式
  lenient —— 不限格式，依次尝试 boxed → answer 行 → 尾部数值
  nopos   —— 敏感性分析：不施加位置约束的尾部数值

规则要点：
  1. 从标记之后取"第一个数值"，允许带单位（"16 miles" → 16），
     但先尝试完整分数（"3/4" → 0.75），避免把 \frac 拆错。
  2. 每个标记取"最后一个能解析出数值的匹配"，避免中间步骤的裸数字被误认。
  3. answer 行同时接受 "Final answer: X"、"**Final answer:** X"、"The final answer is X"。
  4. 尾部数值须落在末尾 20% 字符内；短输出（≤60 字符）不做位置约束。
"""
import argparse
import json
import re
from pathlib import Path

_BOX = re.compile(r"\\boxed\s*\{")
_ANS = re.compile(r"final\s*answer\s*(?:is)?\s*[:\-–]?\s*\**\s*([^\n]{0,60})", re.I)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FRAC = re.compile(r"\(?(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\)?")


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


def _clean(s):
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace(" ", "")
    s = s.replace("\\%", "").rstrip("%").rstrip(".")
    return s


def to_num(s):
    """从一段文本里取数值：先试整串，再试分数，最后取第一个数。"""
    if s is None:
        return None
    t = _clean(s)
    if not t:
        return None
    m = FRAC.fullmatch(t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?", t)
    if m:
        return float(m.group(1))
    m = FRAC.search(t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = NUM.search(t)
    if m:
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def last_boxed(text):
    last = None
    for m in _BOX.finditer(text):
        inner = _match_braces(text, m.end())
        if inner is not None:
            v = to_num(inner)
            if v is not None:
                last = v
    return last


def last_answer_line(text):
    last = None
    for m in _ANS.finditer(text):
        v = to_num(m.group(1))
        if v is not None:
            last = v
    return last


def tail_number(text):
    ms = list(NUM.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    if len(text) > 60 and m.start() < 0.8 * len(text):
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def nopos_number(text):
    ms = list(NUM.finditer(text))
    return float(ms[-1].group(0).replace(",", "")) if ms else None


def extract_strict(text, cond):
    """严格：C2/C5 只认 answer 行，C3/C6 只认 boxed。

    C0/C1/C4 没有格式约束，因此严格 = 宽容（不存在"违规"可罚），
    否则会凭空造出解析税。
    """
    if cond in ("C2", "C5"):
        return last_answer_line(text)
    if cond in ("C3", "C6"):
        return last_boxed(text)
    return extract_lenient(text)


def extract_lenient(text):
    for f in (last_boxed, last_answer_line, tail_number):
        v = f(text)
        if v is not None:
            return v
    return None


def eq(p, g, tol=1e-4):
    return p is not None and abs(p - g) <= tol * max(1.0, abs(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    d = Path(args.run)
    n = 0
    for fp in sorted(d.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        for r in rows:
            cond = r["cond"]
            sp = extract_strict(r["output"], cond)
            lp = extract_lenient(r["output"])
            np_ = nopos_number(r["output"])
            r["strict_pred"] = sp
            r["lenient_pred"] = lp
            r["lenient_nopos_pred"] = np_
            r["strict_ok"] = eq(sp, r["gold"])
            r["lenient_ok"] = eq(lp, r["gold"])
            r["lenient_nopos_ok"] = eq(np_, r["gold"])
        with open(fp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += len(rows)
        sl = sum(r["lenient_ok"] for r in rows) / len(rows)
        ss = sum(r["strict_ok"] for r in rows) / len(rows)
        print(f"{fp.name}: n={len(rows)} lenient={sl:.3f} strict={ss:.3f}")
    print(f"[reextract] {n} 条已重算")


if __name__ == "__main__":
    main()

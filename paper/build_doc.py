"""组装论文 docx：分段插入 markdown，段落之间插图片。"""
import json
import subprocess
import struct
import sys
from pathlib import Path

ED = Path(r"D:\work_buddy\WorkBuddy\resources\app.asar.unpacked\resources\plugins\workbuddy-builtin\skills\workbuddy-local-office-edit\edsdk.py")
PAPER = Path(r"D:\WorkBuddy-Outputs\dase7506\paper")
FIGS = Path(r"D:\WorkBuddy-Outputs\dase7506\figs")
OUT = Path(r"D:\WorkBuddy-Outputs\DASE7506_期末论文_格式的代价.docx")


def call(tool, **kw):
    args = ["python", str(ED), "call", tool, "--json", json.dumps(kw, ensure_ascii=False)]
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(ED.parent))
    out = (r.stdout or "").strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"{tool} 失败: {r.stdout[-400:]} {r.stderr[-400:]}")


def png_size(p):
    with open(p, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def main():
    d = call("create_doc")
    fid = d["file_id"]
    print("doc:", fid)

    pos = 0
    for i in range(1, 5):
        seg = PAPER / f"seg{i}.md"
        r = call("doc_insert_markdown", file_id=fid, idx=pos,
                 markdown=f"file://{seg}")
        pos = r.get("position", r.get("last_edit_index"))
        print(f"seg{i} -> pos {pos}")

        if i <= 3:
            fig = FIGS / f"fig{i}_{['heatmap','pareto','tax'][i-1]}.png"
            w, h = png_size(fig)
            target_w = 470 if i == 1 else 430
            target_h = int(round(h * target_w / w))
            r2 = call("doc_insert_image", file_id=fid, idx=pos,
                      image_path=str(fig), w=target_w, h=target_h)
            print(f"fig{i} -> {target_w}x{target_h}")
            lp = call("doc_get_last_operable_pos", file_id=fid)
            pos = lp["position"]

    r = call("save_file", file_id=fid, file_path=str(OUT))
    print("saved:", r)


if __name__ == "__main__":
    main()

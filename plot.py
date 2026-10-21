#!/usr/bin/env python3
"""plot.py —— L6 结果画图（框架提供）

用法（在 ~/pmpp/ring-allreduce 下，先 make venv）：
  .venv/bin/python plot.py results/allreduce.csv            # → results/busbw.png、results/time.png
  .venv/bin/python plot.py results/k_sweep.csv --k-sweep    # → results/k_sweep.png
选项：
  --nccl-tests PATH   nccl-tests 的 summary.csv（默认 ~/nccl-tests/nccl_matrix/summary.csv，存在时画两条参考虚线）
  --out DIR           输出目录（默认与输入 CSV 同目录）
  --version V         --k-sweep 时画哪个版本（默认 r4）

输入是 ring 写出的长表 CSV：
  version,K,slots,timing,size_bytes,count,time_us,issue_us,algbw_GBps,busbw_GBps,wrong
图中文字用英文，避免服务器上缺中文字体时显示成方块。
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # 服务器上没有显示器
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- 配色与样式
SURFACE = "#fcfcfb"  # 图的底色
INK = "#0b0b0b"      # 标题
INK_2 = "#52514e"    # 轴标签、图例文字
MUTED = "#898781"    # 刻度文字
GRID = "#e1e0d9"     # 网格（细实线）
AXIS = "#c3c2b7"     # 坐标轴

# 每个版本固定一个颜色（按固定顺序分配，与这次出现了哪些版本无关）
SERIES = {
    "r1": "#2a78d6",                      # blue
    "r2": "#eb6834",                      # orange
    "r3": "#1baf7a",                      # aqua
    "r4": "#eda100",                      # yellow
    "r5": "#e87ba4",                      # magenta
    "nccl": "#008300",                    # green
    "nccl-tests Ring_Simple": "#6250d6",  # violet（参考线，虚线）
    "nccl-tests default": "#e34948",      # red（参考线，虚线）
}
# K 扫描里的“尺寸”是有序类别：单一色相、由浅到深
SIZE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
KSWEEP_SIZES = [64 << 10, 1 << 20, 16 << 20, 256 << 20, 1 << 30]

LINE = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round")
REF_LINE = dict(linewidth=2, linestyle=(0, (4, 3)), dash_capstyle="round")


def size_label(b):
    for unit, shift in (("G", 30), ("M", 20), ("K", 10)):
        if b >= 1 << shift and b % (1 << shift) == 0:
            return f"{b >> shift}{unit}"
    return str(b)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def log2_size_axis(ax, sizes):
    """x 轴：消息大小，log2 刻度，每 4× 标一个刻度（1K、4K、16K …）"""
    ax.set_xscale("log", base=2)
    lo, hi = min(sizes), max(sizes)
    ticks, t = [], 1 << 10
    while t <= hi:
        if t >= lo:
            ticks.append(t)
        t *= 4
    ax.set_xticks(ticks)
    ax.set_xticklabels([size_label(t) for t in ticks])
    ax.minorticks_off()


def end_dot(ax, x, y, color):
    """线末端的圆点：8 px，外加 2 px 底色描边，压在其他线上也看得清"""
    ax.plot([x], [y], "o", markersize=8, color=color, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)


def finish(fig, ax, title, subtitle, path):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- 读数据
def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(version=r["version"], K=int(r["K"]), slots=int(r["slots"]),
                             timing=r["timing"], size=int(r["size_bytes"]),
                             time_us=float(r["time_us"]), busbw=float(r["busbw_GBps"]),
                             wrong=int(r["wrong"])))
    bad = sorted({(r["version"], r["size"]) for r in rows if r["wrong"] != 0})
    if bad:
        print(f"警告：{len(bad)} 个（版本, 尺寸）的 #wrong ≠ 0，图里照常画出：{bad[:5]} …")
    return rows


def read_nccl_tests(path):
    """nccl-tests summary.csv 中的 Ring_Simple 与 default_default（out-of-place）"""
    if not path or not os.path.exists(path):
        return {}
    cols = {"nccl-tests Ring_Simple": "Ring_Simple", "nccl-tests default": "default_default"}
    out = {name: [] for name in cols}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for name, col in cols.items():
                bw, t = r.get(col + "_busbw", ""), r.get(col + "_time_us", "")
                if bw and t:
                    out[name].append((int(r["size_bytes"]), float(bw), float(t)))
    return {k: sorted(v) for k, v in out.items() if v}


# ---------------------------------------------------------------- 图
def plot_main(rows, ref, out_dir):
    by_ver = defaultdict(list)
    for r in rows:
        by_ver[r["version"]].append(r)
    versions = [v for v in SERIES if v in by_ver]
    sizes = sorted({r["size"] for r in rows})
    first = rows[0]
    subtitle = (f"4 GPUs, float32 sum, in-place; timing={first['timing']}, "
                f"K={first['K']}, slots={first['slots']}")
    if ref:
        subtitle += "; dashed: nccl-tests (out-of-place, back-to-back)"

    for metric, ylabel, fname, logy in (
            ("busbw", "busbw (GB/s)", "busbw.png", False),
            ("time_us", "time per AllReduce (µs)", "time.png", True)):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        style_axes(ax)
        for v in versions:
            pts = sorted(by_ver[v], key=lambda r: r["size"])
            xs, ys = [r["size"] for r in pts], [r[metric] for r in pts]
            ax.plot(xs, ys, color=SERIES[v], label=v, zorder=3, **LINE)
            end_dot(ax, xs[-1], ys[-1], SERIES[v])
        for name, pts in ref.items():
            pts = [p for p in pts if sizes[0] <= p[0] <= sizes[-1]]
            if not pts:
                continue
            ys = [p[1] if metric == "busbw" else p[2] for p in pts]
            ax.plot([p[0] for p in pts], ys, color=SERIES[name], label=name, zorder=2, **REF_LINE)
        log2_size_axis(ax, sizes)
        ax.set_xlabel("message size per GPU (bytes)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        title = ("AllReduce bus bandwidth vs message size" if metric == "busbw"
                 else "AllReduce time vs message size (the flat part on the left is α)")
        finish(fig, ax, title, subtitle, os.path.join(out_dir, fname))


def plot_ksweep(rows, out_dir, version):
    rows = [r for r in rows if r["version"] == version]
    if not rows:
        print(f"CSV 里没有 {version} 的数据")
        return
    by_size = defaultdict(dict)
    for r in rows:
        by_size[r["size"]][r["K"]] = r["busbw"]
    chosen = [s for s in KSWEEP_SIZES if s in by_size] or sorted(by_size)[:len(SIZE_RAMP)]
    ks = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax)
    for color, s in zip(SIZE_RAMP, chosen):
        pts = sorted(by_size[s].items())
        xs, ys = [k for k, _ in pts], [bw for _, bw in pts]
        ax.plot(xs, ys, color=color, label=size_label(s), zorder=3, **LINE)
        end_dot(ax, xs[-1], ys[-1], color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("K (sub-chunks per chunk)")
    ax.set_ylabel("busbw (GB/s)")
    slots = sorted({r["slots"] for r in rows})
    finish(fig, ax, f"{version}: busbw vs K, one line per message size",
           f"slots={','.join(map(str, slots))}; timing={rows[0]['timing']}",
           os.path.join(out_dir, "k_sweep.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--k-sweep", action="store_true")
    ap.add_argument("--version", default="r4")
    ap.add_argument("--nccl-tests", default=os.path.expanduser("~/nccl-tests/nccl_matrix/summary.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"{args.csv} 里没有数据")
    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    if args.k_sweep:
        plot_ksweep(rows, out_dir, args.version)
    else:
        plot_main(rows, read_nccl_tests(args.nccl_tests), out_dir)


if __name__ == "__main__":
    main()

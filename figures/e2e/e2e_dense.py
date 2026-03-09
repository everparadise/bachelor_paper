#!/usr/bin/env python3
"""
基于 parse_client_jsonl.py 的解析逻辑，对相同 Trace 的不同策略在不同 SC 下的
TTFT 和 TPOT 指标进行折线图比较展示。

输入：manifest CSV 或目录扫描
输出：每行一个 Trace，4 列折线图：TTFT Mean, TTFT P99, TPOT Mean, TPOT P99
横轴：Rate (reqs/sec)，纵轴：时间 (ms)
"""

import csv
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, MaxNLocator

# 使用非交互式后端，便于无 GUI 环境
matplotlib.use("Agg")

# =============================================================================
# 硬编码配置（可直接修改）
# =============================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 数据根目录：脚本所在目录的父目录的父目录下的 data 目录（即项目根/data/）
_data_dir = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "data")

# 硬编码的目录名称（相对于数据根目录）
DEFAULT_ROOT_NAMES = [
    "20260221234355_traceA_e2e",
    "20260220235643_traceB_e2e",
    "20260222184925_coder_e2e",
]
DEFAULT_TIME_IN_SECS = 1200
OUTPUT_FILENAME = "e2e_dense_fig.png"  # 与 e2e_moe 区分，避免覆盖


def resolve_and_extract(root_name: str) -> Optional[str]:
    """
    将硬编码的目录名拼接为完整路径，若对应 .tgz/.tar.gz 存在则先解压。
    参考 parse.py 的解压方式。
    返回解压后的目录路径，若目录或解压后目录不存在则返回 None。
    """
    full_path = os.path.join(_data_dir, root_name)
    archive_tgz = full_path + ".tgz"
    archive_tar_gz = full_path + ".tar.gz"

    # 若已是目录且存在，直接返回
    if os.path.isdir(full_path):
        return full_path

    # 若 .tgz 存在，解压
    if os.path.isfile(archive_tgz):
        print(f"📦 Extracting {archive_tgz} into {_data_dir}...")
        with tarfile.open(archive_tgz, "r:gz") as tar:
            tar.extractall(path=_data_dir)
        print("✅ Extraction done.")
        return full_path if os.path.isdir(full_path) else None

    # 若 .tar.gz 存在，解压
    if os.path.isfile(archive_tar_gz):
        print(f"📦 Extracting {archive_tar_gz} into {_data_dir}...")
        with tarfile.open(archive_tar_gz, "r:gz") as tar:
            tar.extractall(path=_data_dir)
        print("✅ Extraction done.")
        return full_path if os.path.isdir(full_path) else None

    return None


def time_formatter(x, pos):
    """参考 parse_client.py：y 轴时间格式化"""
    if x < 1000:
        return f"{int(x)}ms"
    return f"{x/1000:g}s"

# =============================================================================
# 全局样式配置（参考 parse_client.py，便于修改）
# =============================================================================
# 与 parse_client.py 一致的字体与线宽
fontsize = 7
marker_sz = 2.0
linewidth = 0.5
leg_font = fontsize - 0.5
markeredgewidth_sz = 0.5
xlabel_sz = fontsize - 1
ylabel_sz = fontsize - 1

STRATEGY_COLORS = [
    "#1f77b4",  # 蓝 (Xmetric)
    "#d62728",  # 红 (vLLM-v1)
    "#2ca02c",  # 绿 (Dynamo)
    "#9467bd",  # 紫 (Mooncake)
    "#ff7f0e",  # 橙 (Bailian)
]

# 折线数据点样式：marker, linestyle（markersize/linewidth 用全局变量）
LINE_STYLES = [
    ("o", "-"),
    ("s", "--"),
    ("^", "-."),
    ("x", ":"),
    ("D", "-"),
]

# 分位数（可改为 95 等）
PERCENTILE = 99

# Trace 显示名称（参考 parse_client.py 的 NAME）
TRACE_DISPLAY_NAMES = {
    "traceA": "Chatbot",
    "TraceA": "Chatbot",
    "traceB": "Agent",
    "TraceB": "Agent",
    "coder": "Coder",
    "Coder": "Coder",
}

# 策略显示名称映射
STRATEGY_DISPLAY_NAMES = {
    "bailian-impl-075-deterministic": "Company-X",
    "dynamo-deterministic": "Dynamo",
    "join-shortest-q-ttft": "Predictor",
    "join-shortest-q-weight": "vLLM-v1",
}


def adjust_ax_style(ax, fs=fontsize):
    """淡边框、四边均显示、网格、刻度线向内"""
    ax.tick_params(axis="both", labelsize=fs - 1, direction="in")
    ax.tick_params(axis="y", which="major", pad=3)
    ax.tick_params(axis="x", pad=2)
    for spine in ax.spines.values():
        spine.set_color("0.6")
        spine.set_linewidth(0.3)
    ax.grid(True, linestyle=(0, (5, 10)), linewidth=0.3, color="0.7", zorder=0)


# =============================================================================
# 解析逻辑（与 parse_client_jsonl.py 一致）
# =============================================================================
def is_success_status(status):
    """判断 status 是否为 HTTP 成功状态码 (2xx)"""
    if status is None:
        return False
    try:
        code = int(status)
        return 200 <= code < 300
    except (ValueError, TypeError):
        return False


def parse_numeric(val):
    """解析数值，'nil' 或无效值返回 None"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val < 0:
            return None
        return float(val)
    s = str(val).strip().lower()
    if s in ("nil", "null", "", "nan"):
        return None
    try:
        v = float(s)
        return v if v >= 0 else None
    except ValueError:
        return None


def percentile(arr, p):
    """计算百分位数"""
    if not arr:
        return None
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(arr) else f
    return arr[f] + (k - f) * (arr[c] - arr[f]) if f != c else arr[f]


def parse_jsonl_file(file_path, p=PERCENTILE):
    """
    解析单个 jsonl 文件，返回统计字典。
    与 parse_client_jsonl.py 逻辑一致，分位数 p 可配置。
    返回 None 表示文件不存在或解析失败。
    """
    first_token_times = []
    avg_times_between_tokens = []
    total_count = 0
    success_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_count += 1
                if is_success_status(obj.get("status")):
                    success_count += 1

                ftt = parse_numeric(obj.get("first_token_time"))
                if ftt is not None:
                    first_token_times.append(ftt)

                atbt = parse_numeric(obj.get("avg_time_between_tokens"))
                if atbt is not None:
                    avg_times_between_tokens.append(atbt)
    except (FileNotFoundError, OSError):
        return None

    def compute_stats(values):
        if not values:
            return None, None
        return (
            sum(values) / len(values),
            percentile(values, p),
        )

    ttft_mean, ttft_px = compute_stats(first_token_times)
    tpot_mean, tpot_px = compute_stats(avg_times_between_tokens)

    return {
        "ttft_mean": ttft_mean,
        "ttft_px": ttft_px,
        "tpot_mean": tpot_mean,
        "tpot_px": tpot_px,
        "success_count": success_count,
        "total_count": total_count,
    }


# =============================================================================
# 数据收集
# =============================================================================
def load_manifest(manifest_path, time_in_secs=1200):
    """
    从 CSV manifest 加载数据点。
    CSV 列：trace, strategy, sc, path [, time_in_secs]
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            path = r.get("path", "").strip()
            if not path or not os.path.isfile(path):
                continue
            t = r.get("time_in_secs", str(time_in_secs)).strip()
            t_sec = int(t) if t else time_in_secs
            rows.append({
                "trace": r.get("trace", "").strip(),
                "strategy": r.get("strategy", "").strip(),
                "sc": r.get("sc", "").strip(),
                "path": path,
                "time_in_secs": t_sec,
            })
    return rows


def _extract_trace_from_run_dir(run_dir_name):
    """从 run 目录名提取 trace，只要包含 traceA/traceB/coder 即可（支持 _e2e 等后缀）"""
    base = run_dir_name.lower()
    # 按长度优先匹配，避免 traceB 被 trace 误匹配
    for kw, canonical in [("tracea", "traceA"), ("traceb", "traceB"), ("coder", "coder")]:
        if kw in base:
            return canonical
    return run_dir_name


def scan_directory(root_dir, filename="client_code_1.jsonl", time_in_secs=1200):
    """
    扫描目录结构，支持：
    - {root}/{run_dir}/sc{sc}/{strategy}/client_code_1.jsonl  (run_dir 如 20250101_traceA)
    - {root}/{trace}/sc{sc}/{strategy}/client_code_1.jsonl
    - {root}/sc{sc}/{strategy}/client_code_1.jsonl（单 trace，trace 从 root 名解析）
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        return []

    sc_re = re.compile(r"^sc([\d.]+)$", re.I)
    rows = []

    for dirpath, _dirnames, filenames in os.walk(root):
        if filename not in filenames:
            continue
        fpath = os.path.join(dirpath, filename)
        rel = os.path.relpath(dirpath, root)
        parts = Path(rel).parts

        trace = ""
        sc = ""
        strategy = ""

        if len(parts) >= 3:
            # run_dir/scX.X/strategy 或 trace/scX.X/strategy
            if sc_re.match(parts[1]):
                run_or_trace = parts[0]
                sc = sc_re.match(parts[1]).group(1)
                strategy = parts[2]
                trace = _extract_trace_from_run_dir(run_or_trace)
        elif len(parts) >= 2:
            # scX.X/strategy（root 即为 run 目录）
            if sc_re.match(parts[0]):
                sc = sc_re.match(parts[0]).group(1)
                strategy = parts[1]
                trace = _extract_trace_from_run_dir(root.name)

        if trace and sc and strategy:
            rows.append({
                "trace": trace,
                "strategy": strategy,
                "sc": sc,
                "path": fpath,
                "time_in_secs": time_in_secs,
            })
    return rows


def collect_metrics(rows, p=PERCENTILE):
    """
    对每个数据点解析 jsonl 并计算 rate 与指标。

    Rate 计算：rate = total_count / time_in_secs
    - total_count: jsonl 中的总请求数
    - time_in_secs: 该次运行的时长（秒），默认 1200
    - 即 Rate = 总请求数/秒，单位 reqs/sec
    """
    results = []
    for r in rows:
        stats = parse_jsonl_file(r["path"], p=p)
        if stats is None:
            continue
        t_sec = r.get("time_in_secs", 1200)
        rate = stats["total_count"] / t_sec if t_sec > 0 else 0
        results.append({
            "trace": r["trace"],
            "strategy": r["strategy"],
            "sc": r["sc"],
            "rate": rate,
            "success_count": stats["success_count"],
            "total_count": stats["total_count"],
            "ttft_mean": stats["ttft_mean"],
            "ttft_px": stats["ttft_px"],
            "tpot_mean": stats["tpot_mean"],
            "tpot_px": stats["tpot_px"],
        })
    return results


def output_metrics_table(results, output_path, p=PERCENTILE):
    """
    输出不同策略在不同 Trace 上的性能指标表格。
    相同 Rate、相同 Trace 的不同策略放在一起，rate 取整到十分位。
    格式：Trace, strategy, rate, success_count, total_count, ttft_mean, ttft_p99, tpot_mean, tpot_p99
    字段间用 \\t 分隔，便于比较观察。同时输出到 TSV 文件和命令行。
    """
    base = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
    tsv_path = base + "_metrics.tsv"
    sep = "\t"
    # 为每条记录添加 rate 取整到十分位
    for r in results:
        r["_rate_rounded"] = round(r["rate"], 1)
    # 按 Trace, rate(取整), strategy 排序，使相同 Trace+rate 的策略相邻
    sorted_results = sorted(
        results,
        key=lambda x: (x["trace"], x["_rate_rounded"], x["strategy"]),
    )
    header = ["Trace", "strategy", "rate", "success_count", "total_count", "ttft_mean", f"ttft_p{p}", "tpot_mean", f"tpot_p{p}"]
    lines = []
    lines.append(sep.join(header))
    for r in sorted_results:
        trace_display = TRACE_DISPLAY_NAMES.get(r["trace"], r["trace"])
        strategy_display = STRATEGY_DISPLAY_NAMES.get(r["strategy"], r["strategy"])
        row = [
            trace_display,
            strategy_display,
            f"{r['_rate_rounded']:.1f}",
            str(r.get("success_count", "")),
            str(r.get("total_count", "")),
            f"{r['ttft_mean']:.1f}" if r["ttft_mean"] is not None else "",
            f"{r['ttft_px']:.1f}" if r["ttft_px"] is not None else "",
            f"{r['tpot_mean']:.1f}" if r["tpot_mean"] is not None else "",
            f"{r['tpot_px']:.1f}" if r["tpot_px"] is not None else "",
        ]
        lines.append(sep.join(row))
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"性能指标表已保存: {tsv_path}")
    # 命令行输出完整 TSV 表格
    print("\n--- 统计结果 (TSV 格式) ---")
    print("\n".join(lines))
    # 控制台打印表格（相同 Trace+rate 的策略分组显示）
    print("\n--- 各策略在各 Trace 上的性能表格 (按 Trace, rate 分组) ---")
    from itertools import groupby
    for (trace, rate_r), group in groupby(sorted_results, key=lambda x: (x["trace"], x["_rate_rounded"])):
        trace_display = TRACE_DISPLAY_NAMES.get(trace, trace)
        print(f"\n[{trace_display} | rate={rate_r:.1f}]")
        for r in group:
            strategy_display = STRATEGY_DISPLAY_NAMES.get(r["strategy"], r["strategy"])
            ttft = f"{r['ttft_mean']:.1f}" if r["ttft_mean"] is not None else "-"
            ttft_p = f"{r['ttft_px']:.1f}" if r["ttft_px"] is not None else "-"
            tpot = f"{r['tpot_mean']:.1f}" if r["tpot_mean"] is not None else "-"
            tpot_p = f"{r['tpot_px']:.1f}" if r["tpot_px"] is not None else "-"
            print(f"  {strategy_display}\t{rate_r:.1f}\t{r.get('success_count','')}\t{r.get('total_count','')}\tTTFT={ttft}/{ttft_p}\tTPOT={tpot}/{tpot_p}")


# =============================================================================
# 绘图
# =============================================================================
def plot_comparison(
    results,
    output_path,
    p=PERCENTILE,
    colors=None,
    line_styles=None,
):
    """
    绘制比较图：每行一个 Trace，4 列分别为 TTFT Mean, TTFT P99, TPOT Mean, TPOT P99
    每行最左侧为 Trace 名称，图例在最上方
    """
    colors = colors or STRATEGY_COLORS
    line_styles = line_styles or LINE_STYLES

    # 参考 parse_client.py 的 rc 设置
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["lines.markersize"] = marker_sz

    traces = sorted(set(r["trace"] for r in results))
    strategies = sorted(set(r["strategy"] for r in results))

    n_traces = len(traces)
    n_cols = 4
    metrics = [
        ("ttft_mean", f"TTFT Mean (ms)"),
        ("ttft_px", f"TTFT P{p} (ms)"),
        ("tpot_mean", f"TPOT Mean (ms)"),
        ("tpot_px", f"TPOT P{p} (ms)"),
    ]

    # 使用 GridSpec：宽度 +1/4，高度 -1/3；间距更紧凑
    row_height = 4 * (2 / 3) * (2 / 3) * (2 / 3)
    base_w = 15 * (2 / 3) * (2 / 3)
    fig_w, fig_h = base_w * 1.25, row_height * n_traces
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_traces, 5, width_ratios=[0.8, 1, 1, 1, 1], hspace=0.22, wspace=0.39)
    axes = []
    for row_idx in range(n_traces):
        row_axes = []
        for col_idx in range(5):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
        axes.append(row_axes)

    # 策略 -> 颜色、样式（使用 parse_client.py 的 marker_sz, linewidth）
    strategy_style = {}
    for i, s in enumerate(strategies):
        ls = line_styles[i % len(line_styles)]
        strategy_style[s] = {
            "color": colors[i % len(colors)],
            "marker": ls[0],
            "markersize": marker_sz,
            "linewidth": linewidth,
            "linestyle": ls[1],
        }

    all_rates = [r["rate"] for r in results]
    rate_min = min(all_rates) if all_rates else 0

    all_handles, all_labels = [], []

    for row_idx, trace in enumerate(traces):
        trace_data = [r for r in results if r["trace"] == trace]

        # 最左侧：Trace 名称竖着写，字号 -1，向右移动 0.05，后缀 (Qwen)
        ax_label = axes[row_idx][0]
        trace_display = TRACE_DISPLAY_NAMES.get(trace, trace) + " (Qwen)"
        ax_label.text(0.83, 0.5, trace_display, transform=ax_label.transAxes, fontsize=fontsize - 1, va="center", ha="center", rotation=90)
        ax_label.axis("off")

        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx + 1]

            subplot_ys = []
            for strategy in strategies:
                pts = [(r["rate"], r[metric_key]) for r in trace_data if r["strategy"] == strategy and r[metric_key] is not None]
                if not pts:
                    continue
                pts.sort(key=lambda x: x[0])
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                subplot_ys.extend(ys)
                style = strategy_style[strategy]
                strategy_display = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
                line, = ax.plot(
                    xs, ys,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    markeredgewidth=markeredgewidth_sz,
                    label=strategy_display,
                )
                if strategy_display not in all_labels:
                    all_handles.append(line)
                    all_labels.append(strategy_display)

            # 指标名称移到折线图上侧，向下移动 0.03，背景变宽
            ax.set_title(metric_label, y=0.85, fontsize=xlabel_sz, bbox=dict(boxstyle="round,pad=0", facecolor="white", edgecolor="none"))
            # Rate (reqs/sec) 只在最后一行显示
            if row_idx == n_traces - 1:
                ax.set_xlabel("Rate (reqs/sec)", fontsize=xlabel_sz, labelpad=0)
            else:
                ax.set_xlabel("", fontsize=xlabel_sz)
            # 每行最左侧图像（第一列）的 y 轴左侧添加 "Time"，向左移动约 3 个 y 轴刻度字符
            if col_idx == 0:
                ax.set_ylabel("Time", fontsize=ylabel_sz)
                ax.yaxis.set_label_coords(-0.34, 0.5)
            else:
                ax.set_ylabel("", fontsize=ylabel_sz)
            ax.set_ylim(bottom=0)
            # 修复缩放：按数据范围设置坐标轴，避免折线挤在角落
            trace_rates = [r["rate"] for r in trace_data]
            x_min = min(trace_rates) if trace_rates else rate_min
            x_max = max(trace_rates) if trace_rates else rate_min + 1
            x_range = max(x_max - x_min, 0.01)
            ax.set_xlim(left=x_min - 0.02 * x_range, right=x_max + 0.02 * x_range)
            if subplot_ys:
                # 最高点之上留出 0.05 空间（相对 y 范围），避免名称覆盖折线
                y_max_val = max(subplot_ys)
                y_max = y_max_val * 1.08 + 0.05 * y_max_val
                ax.set_ylim(bottom=0, top=y_max)
            adjust_ax_style(ax, fontsize)
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))

    # 图例：无边框，向上移动 1 字符（图例字号高度）
    fig.legend(
        all_handles, all_labels,
        loc="upper center", ncol=len(strategies),
        fontsize=leg_font + 1,
        frameon=False,
        handlelength=1.6, handletextpad=0.8, labelspacing=0.0,
        bbox_to_anchor=(0.54, 0.97),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    # 使用硬编码路径，输出到脚本同目录下的 e2e_fig.png
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    rows = []
    for name in DEFAULT_ROOT_NAMES:
        root_dir = resolve_and_extract(name)
        if root_dir is None:
            print(f"Warning: 未找到数据目录或归档 {name}，跳过", file=sys.stderr)
            continue
        rows.extend(scan_directory(root_dir, time_in_secs=DEFAULT_TIME_IN_SECS))

    if not rows:
        print("Error: 未找到有效数据点", file=sys.stderr)
        sys.exit(1)

    results = collect_metrics(rows, p=PERCENTILE)
    if not results:
        print("Error: 未能解析任何有效 jsonl", file=sys.stderr)
        sys.exit(1)

    output_metrics_table(results, output_path, p=PERCENTILE)
    plot_comparison(results, output_path, p=PERCENTILE)
    print("Done.")


if __name__ == "__main__":
    main()

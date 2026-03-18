#!/usr/bin/env python3
"""
基于 e2e_moe.py，支持多目录输入、workload/sc/策略名称映射，以及 workload+sc -> 请求速率的自定义映射。

输入：通过 OUTPUT_DIR_ROOT + OUTPUT_DIR_PATH 生成多个目录，每个目录名包含 workload 标识（如 traceA、traceB、coder）
目录结构：{dir}/sc{X.X}/{strategy}/client_code_1.jsonl
输出：每行一个 workload，4 列折线图：TTFT Mean, TTFT P99, TPOT Mean, TPOT P99
横轴：请求速率（来自 WORKLOAD_SC_TO_RATE 映射），纵轴：时间 (ms)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, MaxNLocator

matplotlib.use("Agg")

# =============================================================================
# 全局配置
# =============================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TIME_IN_SECS = 1200
OUTPUT_FILENAME = "e2e_new_fig.png"

OUTPUT_DIR_ROOT = "/nvme/lmetric/logs/lmmetric-logs"

OUTPUT_DIR_PATH = [
    "20260307132922_moe_traceA_jsq",
    "20260308121436_moe_coder_jsq",
    "20260308121436_moe_traceB_jsq",
    "20260309123944_moe_coder_dynamo",
    "20260309140449_moe_traceA_dynamo",
    "20260309140449_moe_traceB_dynamo",
    "20260309195244_moe_coder_bailian",
    "20260309195244_moe_traceA_bailian",
    "20260309195244_moe_traceB_bailian",
    "20260311120407_moe_traceA_vllm",
    "20260312003413_moe_coder_vllm",
    "20260312003413_moe_traceB_vllm",
]

TRACEA_RATE = 5.0
TRACEB_RATE = 24.0
CODER_RATE = 5.0

# workload 显示名称映射（目录名包含的字段 -> 显示名）
WORKLOAD_DISPLAY_NAMES = {
    "traceA": "Chatbot",
    "tracea": "Chatbot",
    "traceB": "Agent",
    "traceb": "Agent",
    "coder": "Coder",
    "Coder": "Coder",
}

# 策略名称映射（目录中的策略名 -> 显示名）
STRATEGY_DISPLAY_NAMES = {
    "bailian-impl-075-deterministic": "Company-X",
    "dynamo-deterministic": "Dynamo",
    "join-shortest-q-ttft": "Predictor",
    "join-shortest-q-weight": "vLLM-v1",
}

# workload -> sc -> 请求速率 (reqs/sec)，用于横坐标
# 可在此修改映射，支持 traceA/traceB/coder 等
WORKLOAD_SC_TO_RATE = {
    "traceA": {"3.6": 3.6 * TRACEA_RATE, "3.8": 3.8 * TRACEA_RATE, "4.0": 4.0 * TRACEA_RATE, "4.2": 4.2 * TRACEA_RATE, "4.4": 4.4 * TRACEA_RATE, "4.6": 4.6 * TRACEA_RATE},
    "traceB": {"3.1": 3.1 * TRACEB_RATE, "3.4": 3.4 * TRACEB_RATE, "3.7": 3.7 * TRACEB_RATE, "4.0": 4.0 * TRACEB_RATE, "4.3": 4.3 * TRACEB_RATE},
    "coder": {"0.9": 0.9 * CODER_RATE, "1.1": 1.1 * CODER_RATE, "1.3": 1.3 * CODER_RATE, "1.5": 1.5 * CODER_RATE, "1.7": 1.7 * CODER_RATE},
}

# =============================================================================
# 样式配置
# =============================================================================
fontsize = 7
marker_sz = 2.0
linewidth = 0.5
leg_font = fontsize - 0.5
markeredgewidth_sz = 0.5
xlabel_sz = fontsize - 1
ylabel_sz = fontsize - 1

STRATEGY_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
]

LINE_STYLES = [
    ("o", "-"), ("s", "--"), ("^", "-."), ("x", ":"), ("D", "-"),
]

# PERCENTILE = 99
PERCENTILE = 95


def time_formatter(x, pos):
    if x < 1000:
        return f"{int(x)}ms"
    return f"{x/1000:g}s"


def adjust_ax_style(ax, fs=fontsize):
    ax.tick_params(axis="both", labelsize=fs - 1, direction="in")
    ax.tick_params(axis="y", which="major", pad=3)
    ax.tick_params(axis="x", pad=2)
    for spine in ax.spines.values():
        spine.set_color("0.6")
        spine.set_linewidth(0.3)
    ax.grid(True, linestyle=(0, (5, 10)), linewidth=0.3, color="0.7", zorder=0)


# =============================================================================
# 解析逻辑
# =============================================================================
def is_success_status(status):
    if status is None:
        return False
    try:
        code = int(status)
        return 200 <= code < 300
    except (ValueError, TypeError):
        return False


def parse_numeric(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val >= 0 else None
    s = str(val).strip().lower()
    if s in ("nil", "null", "", "nan"):
        return None
    try:
        v = float(s)
        return v if v >= 0 else None
    except ValueError:
        return None


def percentile(arr, p):
    if not arr:
        return None
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(arr) else f
    return arr[f] + (k - f) * (arr[c] - arr[f]) if f != c else arr[f]


def parse_jsonl_file(file_path, p=PERCENTILE):
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
        return sum(values) / len(values), percentile(values, p)

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
# 目录扫描与 workload 提取
# =============================================================================
def extract_workload_from_dirname(dir_name):
    """从目录名提取 workload，匹配 traceA/traceB/coder 等"""
    base = dir_name
    for kw, canonical in [
        ("tracea", "traceA"),
        ("traceb", "traceB"),
        ("coder", "coder"),
    ]:
        if kw in base.lower():
            return canonical
    return dir_name


def scan_directory(root_dir, workload_override=None, filename="client_code_1.jsonl", time_in_secs=1200):
    """
    扫描目录结构，支持：
    - {root}/sc{X.X}/{strategy}/client_code_1.jsonl
    - {root}/{run_dir}/sc{X.X}/{strategy}/client_code_1.jsonl
    若 workload_override 为 None，则从 root 或 run_dir 目录名解析 workload。
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

        workload = workload_override
        sc = ""
        strategy = ""

        if len(parts) >= 3:
            if sc_re.match(parts[1]):
                run_or_trace = parts[0]
                sc = sc_re.match(parts[1]).group(1)
                strategy = parts[2]
                workload = workload or extract_workload_from_dirname(run_or_trace)
        elif len(parts) >= 2:
            if sc_re.match(parts[0]):
                sc = sc_re.match(parts[0]).group(1)
                strategy = parts[1]
                workload = workload or extract_workload_from_dirname(root.name)

        if workload and sc and strategy:
            rows.append({
                "workload": workload,
                "strategy": strategy,
                "sc": sc,
                "path": fpath,
                "time_in_secs": time_in_secs,
            })

    return rows


def get_rate_for_workload_sc(workload, sc):
    """从 WORKLOAD_SC_TO_RATE 获取速率；若无映射返回 None"""
    w = WORKLOAD_SC_TO_RATE.get(workload, {})
    rate = w.get(sc)
    if rate is not None:
        return float(rate)
    # 尝试归一化 sc（如 "3" -> "3.0"）再查找
    try:
        sc_norm = f"{float(sc):.1f}" if "." not in str(sc) else sc
        rate = w.get(sc_norm)
        if rate is not None:
            return float(rate)
    except (ValueError, TypeError):
        pass
    return None


# =============================================================================
# 数据收集
# =============================================================================
def collect_metrics(rows, p=PERCENTILE):
    results = []
    for r in rows:
        stats = parse_jsonl_file(r["path"], p=p)
        if stats is None:
            continue
        rate = get_rate_for_workload_sc(r["workload"], r["sc"])
        if rate is None:
            print(
                f'Warning: 未找到 workload/sc 对应的 rate 映射，跳过: workload={r["workload"]}, sc={r["sc"]}',
                file=sys.stderr,
            )
            continue

        results.append({
            "workload": r["workload"],
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


def output_points_table(point_rows, output_path):
    """
    输出每个折线图中每条线每个点的坐标到 TSV。
    字段：workload, metric, strategy, sc, x_rate, y_value
    """
    base = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
    tsv_path = base + "_points.tsv"
    header = ["workload", "metric", "strategy", "sc", "x_rate", "y_value"]
    lines = ["\t".join(header)]

    for row in point_rows:
        lines.append(
            "\t".join(
                [
                    str(row["workload"]),
                    str(row["metric"]),
                    str(row["strategy"]),
                    str(row["sc"]),
                    f'{row["x_rate"]:.6g}',
                    f'{row["y_value"]:.6g}',
                ]
            )
        )

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Point coordinates saved: {tsv_path}")


# =============================================================================
# 绘图
# =============================================================================
def plot_comparison(results, output_path, p=PERCENTILE):
    """
    绘制比较图：每行一个 workload，4 列分别为 TTFT Mean, TTFT P99, TPOT Mean, TPOT P99。
    相同 workload、相同 sc 的不同策略绘制在同一张图中（不同策略为不同折线）。
    横坐标使用 WORKLOAD_SC_TO_RATE 映射的请求速率。
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["lines.markersize"] = marker_sz

    workloads = sorted(set(r["workload"] for r in results))
    strategies = sorted(set(r["strategy"] for r in results))

    n_workloads = len(workloads)
    metrics = [
        ("ttft_mean", f"TTFT Mean (ms)"),
        ("ttft_px", f"TTFT P{p} (ms)"),
        ("tpot_mean", f"TPOT Mean (ms)"),
        ("tpot_px", f"TPOT P{p} (ms)"),
    ]

    row_height = 4 * (2 / 3) ** 3
    base_w = 15 * (2 / 3) ** 2
    fig_w, fig_h = base_w * 1.25, row_height * n_workloads
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_workloads, 5, width_ratios=[0.8, 1, 1, 1, 1], hspace=0.22, wspace=0.39)
    axes = []

    strategy_style = {}
    for i, s in enumerate(strategies):
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        strategy_style[s] = {
            "color": STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
            "marker": ls[0],
            "markersize": marker_sz,
            "linewidth": linewidth,
            "linestyle": ls[1],
        }

    all_rates = [r["rate"] for r in results]
    rate_min = min(all_rates) if all_rates else 0

    all_handles, all_labels = [], []
    point_rows = []

    for row_idx, workload in enumerate(workloads):
        workload_data = [r for r in results if r["workload"] == workload]
        workload_rates = sorted(set(r["rate"] for r in workload_data if r["rate"] is not None))
        row_axes = []
        for col_idx in range(5):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
        axes.append(row_axes)

        ax_label = axes[row_idx][0]
        workload_display = WORKLOAD_DISPLAY_NAMES.get(workload, workload) + " (Qwen)"
        ax_label.text(0.83, 0.5, workload_display, transform=ax_label.transAxes,
                     fontsize=fontsize - 1, va="center", ha="center", rotation=90)
        ax_label.axis("off")

        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx + 1]

            subplot_ys = []
            for strategy in strategies:
                pts = [
                    (r["rate"], r[metric_key], r["sc"])
                    for r in workload_data
                    if r["strategy"] == strategy and r[metric_key] is not None and r["rate"] is not None
                ]
                if not pts:
                    continue
                pts.sort(key=lambda x: (x[0], float(x[2]) if str(x[2]).replace(".", "", 1).isdigit() else str(x[2])))
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

                for x_val, y_val, sc_val in pts:
                    point_rows.append(
                        {
                            "workload": workload,
                            "metric": metric_key,
                            "strategy": strategy_display,
                            "sc": sc_val,
                            "x_rate": x_val,
                            "y_value": y_val,
                        }
                    )

            ax.set_title(metric_label, y=0.85, fontsize=xlabel_sz,
                        bbox=dict(boxstyle="round,pad=0", facecolor="white", edgecolor="none"))
            if row_idx == n_workloads - 1:
                ax.set_xlabel("Rate (reqs/sec)", fontsize=xlabel_sz, labelpad=0)
            else:
                ax.set_xlabel("", fontsize=xlabel_sz)
            if col_idx == 0:
                ax.set_ylabel("Time", fontsize=ylabel_sz)
                ax.yaxis.set_label_coords(-0.34, 0.5)
            else:
                ax.set_ylabel("", fontsize=ylabel_sz)
            ax.set_ylim(bottom=0)

            trace_rates = [r["rate"] for r in workload_data]
            if workload_rates:
                x_min = min(workload_rates)
                x_max = max(workload_rates)
            else:
                x_min = min(trace_rates) if trace_rates else rate_min
                x_max = max(trace_rates) if trace_rates else rate_min + 1
            x_span = max(x_max - x_min, 1.0)
            x_pad = max(0.06 * x_span, 0.3)
            ax.set_xlim(left=x_min - x_pad, right=x_max + x_pad)
            if workload_rates:
                ax.set_xticks(workload_rates)
                ax.set_xticklabels([f"{v:.1f}" for v in workload_rates])
            if subplot_ys:
                y_max_val = max(subplot_ys)
                y_max = y_max_val * 1.08 + 0.05 * y_max_val
                ax.set_ylim(bottom=0, top=y_max)
            adjust_ax_style(ax, fontsize)
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))
            if not workload_rates:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))

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
    return point_rows


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="多目录 e2e 性能比较图，支持 workload/sc/策略映射"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(_script_dir, OUTPUT_FILENAME),
        help="输出图片路径",
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=DEFAULT_TIME_IN_SECS,
        help="单次运行时长（秒），保留目录扫描兼容参数",
    )
    args = parser.parse_args()

    input_dirs = [
        os.path.join(OUTPUT_DIR_ROOT, dir_name)
        for dir_name in OUTPUT_DIR_PATH
    ]

    rows = []
    for d in input_dirs:
        if not os.path.isdir(d):
            print(f"Warning: 目录不存在，跳过: {d}", file=sys.stderr)
            continue
        rows.extend(scan_directory(d, time_in_secs=args.time))

    if not rows:
        print("Error: 未找到有效数据点", file=sys.stderr)
        sys.exit(1)

    results = collect_metrics(rows, p=PERCENTILE)
    if not results:
        print("Error: 未能解析任何有效 jsonl", file=sys.stderr)
        sys.exit(1)

    point_rows = plot_comparison(results, args.output, p=PERCENTILE)
    output_points_table(point_rows, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

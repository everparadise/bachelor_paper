#!/usr/bin/env python3
"""
对给定目录下所有子目录的 client_code_1.jsonl 统计 TTFT Mean, TTFT P95, TPOT Mean, TPOT P95，
并用 CDF 图与折线图展示。解析逻辑参考 parse_client_jsonl.py。
"""

import argparse
import json
import os
import re
import sys
import tarfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# =============================================================================
# 硬编码配置
# =============================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 数据根目录：脚本所在目录的父目录的父目录下的 data 目录（即项目根/data/）
_data_dir = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "data")

# 目标目录名称（相对于数据根目录），若为 .tgz/.tar.gz 则自动解压
TARGET_ROOT_NAME = "20260220235643_sc2.8_traceB_bailian_searcher"  # 在此修改为实际目录名

# 折线颜色与数据点样式（参考 e2e_moe.py）
CHART_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # 蓝 红 绿 紫
CHART_MARKERS = ["o", "s", "^", "D"]  # 对应 TTFT Mean, TTFT P95, TPOT Mean, TPOT P95


def resolve_and_extract(root_name: str) -> Optional[str]:
    """
    将硬编码的目录名拼接为完整路径，若对应 .tgz/.tar.gz 存在则先解压。
    参考 figures/e2e/e2e_moe.py 的解压方式。
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


# -----------------------------------------------------------------------------
# 数据点过滤模式（可通过 CLI 覆盖）
# -----------------------------------------------------------------------------
# 模式1: LAMBDA_POINTS=None - 绘制所有数据点
# 模式2: LAMBDA_POINTS=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] - 仅绘制指定 λ 的点
# 模式3: EXCLUDE_LAMBDA_1=True - 绘制全部但排除 λ=1.0（极值会拉远图像）
# LAMBDA_POINTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDA_POINTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXCLUDE_LAMBDA_1 = False


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


def parse_jsonl_with_raw(file_path):
    """
    解析单个 jsonl 文件，返回原始值数组和统计。
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
        return sum(values) / len(values), percentile(values, 95)

    ttft_mean, ttft_p95 = compute_stats(first_token_times)
    tpot_mean, tpot_p95 = compute_stats(avg_times_between_tokens)

    return {
        "ttft_values": first_token_times,
        "tpot_values": avg_times_between_tokens,
        "ttft_mean": ttft_mean,
        "ttft_p95": ttft_p95,
        "tpot_mean": tpot_mean,
        "tpot_p95": tpot_p95,
        "success_count": success_count,
        "total_count": total_count,
    }


def extract_lambda_from_path(path):
    """
    从路径/子目录名中提取 bailian 策略的 λ 参数。
    例如: bailian-impl-085-deterministic -> 0.85, bailian-impl-04-deterministic -> 0.4
    若无法解析则返回 None。
    """
    for name in path.replace("\\", "/").split("/"):
        m = re.search(r"bailian-impl-(\d+)-deterministic", name)
        if m:
            digits = m.group(1)
            n = int(digits)
            if len(digits) == 2:  # 04 -> 0.4, 05 -> 0.5
                return n / 10
            if len(digits) == 3:  # 085 -> 0.85, 045 -> 0.45
                return n / 100
            return n / (10 ** len(digits))
    return None


def find_client_code_jsonl(root_dir, filename="client_code_1.jsonl"):
    """递归查找目录及子目录下所有指定文件"""
    found = []
    root = os.path.abspath(root_dir)
    for dirpath, _dirnames, filenames in os.walk(root):
        if filename in filenames:
            found.append(os.path.join(dirpath, filename))
    return sorted(found)


def collect_data(root_dir):
    """收集所有 client_code_1.jsonl 的解析结果"""
    root = os.path.abspath(root_dir)
    files = find_client_code_jsonl(root)
    results = []
    for fpath in files:
        data = parse_jsonl_with_raw(fpath)
        if data is None or (not data["ttft_values"] and not data["tpot_values"]):
            continue
        rel_dir = os.path.relpath(os.path.dirname(fpath), root)
        if rel_dir == ".":
            rel_dir = "(根目录)"
        results.append({"path": rel_dir, "abspath": fpath, **data})
    return results


def _lambda_matches(lam, targets, tol=0.01):
    """lam 是否匹配 targets 中任一值（容差 tol）"""
    if lam is None:
        return False
    return any(abs(lam - t) < tol for t in targets)


def plot_line_charts(results, output_dir):
    """绘制 TTFT Mean/P95 和 TPOT Mean/P95 的折线图。若子目录名为 bailian-impl-XXX-deterministic，则 x 轴为 λ 参数，刻度粒度 0.2"""
    if not results:
        return

    # 尝试提取 λ，用于 x 轴
    with_lambda_raw = [(r, extract_lambda_from_path(r["path"])) for r in results]
    with_lambda = [(r, lam) for r, lam in with_lambda_raw if lam is not None]
    use_lambda = (
        len(with_lambda) > 0
        and all(lam is not None for _, lam in with_lambda_raw)
    )

    if use_lambda:
        # 按 λ 排序
        with_lambda.sort(key=lambda t: t[1])
        results_sorted = [r for r, _ in with_lambda]
        x_vals = np.array([lam for _, lam in with_lambda])
        x_label = "λ"
    else:
        results_sorted = results
        x_vals = np.arange(len(results))
        x_label = None

    ttft_mean = [r["ttft_mean"] if r["ttft_mean"] is not None else float("nan") for r in results_sorted]
    ttft_p95 = [r["ttft_p95"] if r["ttft_p95"] is not None else float("nan") for r in results_sorted]
    tpot_mean = [r["tpot_mean"] if r["tpot_mean"] is not None else float("nan") for r in results_sorted]
    tpot_p95 = [r["tpot_p95"] if r["tpot_p95"] is not None else float("nan") for r in results_sorted]

    def shorten(s, max_len=25):
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    labels = [shorten(r["path"]) for r in results_sorted]

    # 使用 GridSpec：TTFT Mean 和 TTFT P95 使用 broken axis（仅 use_lambda 时），TPOT 正常
    fig = plt.figure(figsize=(14, 10))
    # TTFT broken axis 总高度与 TPOT 单行相同；row 2 为 TTFT 与 TPOT 之间的间距
    n_rows = 4 if use_lambda else 2
    hr = [0.5, 0.5, 0.15, 1] if use_lambda else [1, 1]
    gs = GridSpec(n_rows, 2, figure=fig, height_ratios=hr, hspace=0.06)

    # 断点标记样式（参考 broken.py）
    d = 0.5
    break_kw = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color="k", mec="k", mew=1, clip_on=False)

    def _adjust_ax_style(ax):
        """刻度线向内（direction='in'）"""
        ax.tick_params(axis="both", direction="in")

    def _setup_ax_common(ax, y_vals, color, marker, title):
        """通用：绘制折线并设置 x 轴、标题、网格"""
        ax.plot(x_vals, y_vals, marker + "-", color=color, linewidth=2, markersize=6)
        if use_lambda:
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_xlim(-0.05, 1.05)
        else:
            ax.set_xticks(x_vals)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        if title:
            # ax.set_title(metric_label, y=0.85, fontsize=xlabel_sz, bbox=dict(boxstyle="round,pad=0", facecolor="white", edgecolor="none"))
            ax.set_title(title, y=0.96,bbox=dict(boxstyle="round,pad=0", facecolor="white", edgecolor="none"))
        ax.grid(True, alpha=0.3)
        _adjust_ax_style(ax)

    if use_lambda:
        # TTFT Mean: broken axis，断开 400-900，下 250-400，上 ~1000
        ax_ttft_mean_top = fig.add_subplot(gs[0, 0])
        ax_ttft_mean_bot = fig.add_subplot(gs[1, 0], sharex=ax_ttft_mean_top)
        _setup_ax_common(ax_ttft_mean_top, ttft_mean, CHART_COLORS[0], CHART_MARKERS[0], "TTFT Mean")
        _setup_ax_common(ax_ttft_mean_bot, ttft_mean, CHART_COLORS[0], CHART_MARKERS[0], "")
        ax_ttft_mean_top.set_ylim(1000, 1090)
        ax_ttft_mean_top.spines.bottom.set_visible(False)
        ax_ttft_mean_top.xaxis.tick_top()
        ax_ttft_mean_top.tick_params(labeltop=False)
        ax_ttft_mean_top.set_ylabel("")
        ax_ttft_mean_bot.set_ylim(250, 480)
        ax_ttft_mean_bot.spines.top.set_visible(False)
        ax_ttft_mean_bot.set_xlabel("λ")
        
        # 修改：在底部子图设置 ylabel，并调整位置使其在左侧居中（相对于上下两个子图）
        ax_ttft_mean_bot.set_ylabel("ms")
        # 使用 label_coords 调整位置，x=-0.22 向左移动，y=1.5 向上移动到两个子图的中间位置
        ax_ttft_mean_bot.yaxis.set_label_coords(-0.10, 1.1)
        
        ax_ttft_mean_top.plot([0, 1], [0, 0], transform=ax_ttft_mean_top.transAxes, **break_kw)
        ax_ttft_mean_bot.plot([0, 1], [1, 1], transform=ax_ttft_mean_bot.transAxes, **break_kw)
        plt.setp(ax_ttft_mean_top.get_xticklabels(), visible=False)

        # TTFT P95: broken axis，断开 1600-6000，下 1000-1500，上 ~6131
        ax_ttft_p95_top = fig.add_subplot(gs[0, 1])
        ax_ttft_p95_bot = fig.add_subplot(gs[1, 1], sharex=ax_ttft_p95_top)
        _setup_ax_common(ax_ttft_p95_top, ttft_p95, CHART_COLORS[1], CHART_MARKERS[1], "TTFT P95")
        _setup_ax_common(ax_ttft_p95_bot, ttft_p95, CHART_COLORS[1], CHART_MARKERS[1], "")
        ax_ttft_p95_top.set_ylim(6000, 6190)
        ax_ttft_p95_top.spines.bottom.set_visible(False)
        ax_ttft_p95_top.xaxis.tick_top()
        ax_ttft_p95_top.tick_params(labeltop=False)
        ax_ttft_p95_top.set_ylabel("")
        ax_ttft_p95_bot.set_ylim(1000, 1650)
        ax_ttft_p95_bot.spines.top.set_visible(False)
        ax_ttft_p95_bot.set_xlabel("λ")
        
        # 修改：在底部子图设置 ylabel，并调整位置使其在左侧居中（相对于上下两个子图）
        ax_ttft_p95_bot.set_ylabel("ms")
        # 使用相同的坐标调整，x=-0.22 向左移动，y=1.5 向上移动到两个子图的中间位置
        ax_ttft_p95_bot.yaxis.set_label_coords(-0.10, 1.1)
        
        ax_ttft_p95_top.plot([0, 1], [0, 0], transform=ax_ttft_p95_top.transAxes, **break_kw)
        ax_ttft_p95_bot.plot([0, 1], [1, 1], transform=ax_ttft_p95_bot.transAxes, **break_kw)
        plt.setp(ax_ttft_p95_top.get_xticklabels(), visible=False)
    else:
        # 非 λ 模式：TTFT 使用普通单图，2x2 布局
        ax_ttft_mean = fig.add_subplot(gs[0, 0])
        ax_ttft_p95 = fig.add_subplot(gs[0, 1])
        _setup_ax_common(ax_ttft_mean, ttft_mean, CHART_COLORS[0], CHART_MARKERS[0], "TTFT Mean")
        _setup_ax_common(ax_ttft_p95, ttft_p95, CHART_COLORS[1], CHART_MARKERS[1], "TTFT P95")
        ax_ttft_mean.set_ylabel("ms")
        ax_ttft_p95.set_ylabel("ms")

    # TPOT Mean 和 TPOT P95（正常，无 broken axis）
    tpot_row = 3 if use_lambda else 1
    ax_tpot_mean = fig.add_subplot(gs[tpot_row, 0])
    ax_tpot_p95 = fig.add_subplot(gs[tpot_row, 1])
    _setup_ax_common(ax_tpot_mean, tpot_mean, CHART_COLORS[2], CHART_MARKERS[2], "TPOT Mean")
    _setup_ax_common(ax_tpot_p95, tpot_p95, CHART_COLORS[3], CHART_MARKERS[3], "TPOT P95")
    ax_tpot_mean.set_ylabel("ms")
    ax_tpot_p95.set_ylabel("ms")
    ax_tpot_mean.set_xlabel("λ" if use_lambda else "")
    ax_tpot_p95.set_xlabel("λ" if use_lambda else "")
    ax_tpot_mean.yaxis.set_label_coords(-0.10, 0.5)
    ax_tpot_p95.yaxis.set_label_coords(-0.10, 0.5)

    # fig.suptitle("TTFT / TPOT Comparison", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.06 if use_lambda else 0.3)
    out = os.path.join(output_dir, "metrics_line.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"折线图已保存: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="对目录下所有 client_code_1.jsonl 统计 TTFT/TPOT 并用 CDF 与折线图展示"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=None,
        help="根目录（可选），将递归查找其下所有 client_code_1.jsonl；未指定时使用全局变量 TARGET_ROOT_NAME",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出图片目录，默认与脚本所在目录相同",
    )
    parser.add_argument(
        "--lambda-points",
        nargs="+",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="模式2: 仅绘制指定 λ 的数据点，如 --lambda-points 0.0 0.2 0.4 0.6 0.8 1.0",
    )
    parser.add_argument(
        "--exclude-lambda-1",
        action="store_true",
        help="模式3: 排除 λ=1.0 的点（极值会拉远图像）",
    )
    args = parser.parse_args()

    if args.dir:
        root_dir = os.path.abspath(args.dir)
    else:
        root_dir = resolve_and_extract(TARGET_ROOT_NAME)
        if root_dir is None:
            print(f"Error: 未找到数据目录或归档 {TARGET_ROOT_NAME}（数据根: {_data_dir}）", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(root_dir):
        print(f"Error: 不是有效目录: {root_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else _script_dir
    os.makedirs(output_dir, exist_ok=True)

    results = collect_data(root_dir)
    if not results:
        print("Error: 未找到有效的 client_code_1.jsonl", file=sys.stderr)
        sys.exit(1)

    # 数据点过滤：模式1 全部 / 模式2 仅指定 λ / 模式3 排除 λ=1.0
    lambda_points = args.lambda_points if args.lambda_points is not None else LAMBDA_POINTS
    exclude_1 = args.exclude_lambda_1 or EXCLUDE_LAMBDA_1

    if lambda_points is not None and len(lambda_points) > 0:
        results = [r for r in results if _lambda_matches(extract_lambda_from_path(r["path"]), lambda_points)]
        if exclude_1:
            results = [r for r in results if (lam := extract_lambda_from_path(r["path"])) is None or lam < 0.999]
        if not results:
            print("Error: 指定 λ 点后无匹配数据", file=sys.stderr)
            sys.exit(1)
        msg = f"共解析 {len(results)} 个子目录（仅 λ ∈ {lambda_points}"
        if exclude_1:
            msg += "，已排除 λ=1.0"
        print(msg + "）")
    else:
        if exclude_1:
            results = [r for r in results if (lam := extract_lambda_from_path(r["path"])) is None or lam < 0.999]
            if not results:
                print("Error: 排除 λ=1.0 后无有效数据", file=sys.stderr)
                sys.exit(1)
            print(f"共解析 {len(results)} 个子目录（已排除 λ=1.0）")
        else:
            print(f"共解析 {len(results)} 个子目录")
    plot_line_charts(results, output_dir)


if __name__ == "__main__":
    main()
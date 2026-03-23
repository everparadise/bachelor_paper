import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# =============================================================================
# 全局配置
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = "comparison.png"
DETAIL_FILENAME = "ttft_error_details.csv"
ERROR_RATIO_THRESHOLD = 0.3

INPUT_DIRS = [
    "/nvme/lmetric/logs/lmmetric-logs/20260321223708_qwen25_fullargs_TraceB/sc3.1/join-shortest-q-ttft",
    "/nvme/lmetric/logs/lmmetric-logs/20260308121436_moe_traceB_jsq/sc3.1/join-shortest-q-ttft",
]

LEGEND_NAMES = [
    "Qwen2.5-7B",
    "Qwen3-30B-A3B",
]

CURVE_COLORS = [
    "#1f77b4",
    "#d62728",
]

CLIENT_FILENAME = "client_code_1.jsonl"
LOG_FILENAME = "router_v2.log"


def parse_predicted_ttft(log_lines):
    """解析日志中的预测 TTFT，只保留实际被指派实例对应的预测。"""
    pred_pattern = re.compile(
        r"Request_(\d+)\s+estimated ttft:\s*([0-9.]+)\s*ms.*?on\s+Vllm#(\d+)"
    )
    assign_pattern = re.compile(r"Assigning Request_(\d+)\s+to Replica#(\d+)")

    preds_all = {}
    assigned_replica = {}

    for line in log_lines:
        m_pred = pred_pattern.search(line)
        if m_pred:
            rid = int(m_pred.group(1))
            ttft = float(m_pred.group(2))
            replica_id = int(m_pred.group(3))
            preds_all[(rid, replica_id)] = ttft
            continue

        m_assign = assign_pattern.search(line)
        if m_assign:
            rid = int(m_assign.group(1))
            replica_id = int(m_assign.group(2))
            assigned_replica[rid] = replica_id

    preds = {}
    missing = 0
    for rid, replica_id in assigned_replica.items():
        key = (rid, replica_id)
        if key in preds_all:
            preds[rid] = preds_all[key]
        else:
            missing += 1

    if not assigned_replica:
        for (rid, _replica_id), ttft in preds_all.items():
            preds[rid] = ttft

    print(f"✅ 解析到预测行数量（含所有实例）：{len(preds_all)}")
    print(f"✅ 解析到指派行数量：{len(assigned_replica)}")
    print(f"✅ 成功匹配到已指派实例预测条目：{len(preds)}")
    if missing > 0:
        print(f"⚠ 有 {missing} 个请求找不到对应实例的预测记录。")

    return preds


def parse_real_ttft_from_jsonl(path):
    """从 client.jsonl 读取真实 TTFT。"""
    reals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "request_id" not in obj or "first_token_time" not in obj:
                continue

            try:
                rid = int(obj["request_id"])
                real_ttft = float(obj["first_token_time"])
            except (TypeError, ValueError):
                continue
            if real_ttft <= 0:
                continue
            reals[rid] = real_ttft
    return reals


def collect_error_ratios(preds, reals):
    """返回匹配请求的绝对相对误差列表和明细表。"""
    error_ratios = []
    table_rows = []

    print("\n=== TTFT 对比明细（单位：ms） ===")
    print(
        f"{'request_id':>10} | {'pred(ms)':>10} | {'real(ms)':>10} | "
        f"{'diff(ms)':>10} | {'error_ratio':>12}"
    )
    print("-" * 60)

    for rid, real_val in sorted(reals.items()):
        if rid not in preds:
            continue

        pred_val = preds[rid]
        diff = pred_val - real_val
        error_ratio = abs(diff) / real_val
        error_ratios.append(error_ratio)
        table_rows.append((rid, pred_val, real_val, diff, error_ratio))

        print(
            f"{rid:10d} | {pred_val:10.2f} | {real_val:10.2f} | "
            f"{diff:10.2f} | {error_ratio:12.4f}"
        )

    return error_ratios, table_rows


def load_one_series(root_dir):
    root = os.path.abspath(root_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"不是有效目录: {root}")

    client_path = os.path.join(root, CLIENT_FILENAME)
    log_path = os.path.join(root, LOG_FILENAME)

    if not os.path.isfile(client_path):
        raise FileNotFoundError(f"未找到 {client_path}")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"未找到 {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        sim_lines = f.readlines()

    preds = parse_predicted_ttft(sim_lines)
    reals = parse_real_ttft_from_jsonl(client_path)

    print(f"✅ 提取到预测TTFT条目: {len(preds)}")
    print(f"✅ 提取到真实TTFT条目: {len(reals)}")

    matched = set(preds.keys()) & set(reals.keys())
    print(f"✅ 匹配成功的 request_id 数量: {len(matched)}")

    error_ratios, table_rows = collect_error_ratios(preds, reals)
    if not error_ratios:
        raise ValueError(f"目录 {root} 中未找到可用于绘图的匹配请求。")

    return {
        "root": root,
        "error_ratios": np.array(sorted(error_ratios)),
        "table_rows": table_rows,
    }


def report_threshold_ratio(series_list, threshold):
    """输出误差低于阈值的请求比例。"""
    print(f"\n=== 误差阈值统计（|pred-real|/real < {threshold:.0%}） ===")
    for series in series_list:
        errors = series["error_ratios"]
        qualified = int(np.sum(errors < threshold))
        total = len(errors)
        ratio = qualified / total if total else 0.0
        print(
            f"{series['label']}: {qualified}/{total} = {ratio:.2%} "
            f"的请求预测误差小于 {threshold:.0%}"
        )


def save_detail_csv(series_list, output_dir):
    out_csv = os.path.join(output_dir, DETAIL_FILENAME)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("series,request_id,pred_ms,real_ms,diff_ms,error_ratio\n")
        for series in series_list:
            for rid, pred_ms, real_ms, diff_ms, error_ratio in series["table_rows"]:
                f.write(
                    f"{series['label']},{rid},{pred_ms:.4f},{real_ms:.4f},"
                    f"{diff_ms:.4f},{error_ratio:.6f}\n"
                )
    print(f"CSV 已保存: {out_csv}")


def plot_error_cdf(series_list, output_dir):
    plt.figure(figsize=(8, 5))

    for idx, series in enumerate(series_list):
        errors = series["error_ratios"]
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        plt.plot(
            errors,
            cdf,
            color=series["color"],
            linewidth=1.5,
            label=series["label"],
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Error ratio (|pred - real| / real)")
    plt.ylabel("Cumulative Probability")
    plt.title("Qwen2.5-7B与Qwen3-30B-A3B预测准确误差CDF")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(output_dir, OUTPUT_FILENAME)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图片已保存: {out_png}")


def validate_config():
    if len(INPUT_DIRS) != 2:
        raise ValueError("INPUT_DIRS 必须恰好包含两个目录。")
    if len(LEGEND_NAMES) != len(INPUT_DIRS):
        raise ValueError("LEGEND_NAMES 数量必须与 INPUT_DIRS 一致。")
    if len(CURVE_COLORS) < len(INPUT_DIRS):
        raise ValueError("CURVE_COLORS 数量不足。")
    for idx, path in enumerate(INPUT_DIRS):
        if not path:
            raise ValueError(f"INPUT_DIRS[{idx}] 为空，请先填入目录路径。")


def main():
    validate_config()

    series_list = []
    for idx, root_dir in enumerate(INPUT_DIRS):
        print(f"\n=== 处理数据目录: {root_dir} ===")
        series = load_one_series(root_dir)
        series["label"] = LEGEND_NAMES[idx]
        series["color"] = CURVE_COLORS[idx]
        series_list.append(series)

    output_dir = SCRIPT_DIR
    plot_error_cdf(series_list, output_dir)
    save_detail_csv(series_list, output_dir)
    report_threshold_ratio(series_list, ERROR_RATIO_THRESHOLD)


if __name__ == "__main__":
    main()

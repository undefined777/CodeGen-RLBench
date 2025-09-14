"""
PPO KL variants (eos/dis) x (reward/loss) on one plot.
- Inputs: TensorBoard event dirs and scalar tag names.
- Output: ppo_kl_variants_<TASK>.png / .pdf
- Style: matplotlib only; one chart; no manual colors; raw (faint) + EMA (solid).
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ====== <<< 填这里：任务名、事件目录、tag 名 >>> ======
TASK_NAME = "c++_to_python"  # 仅用于文件名/标题

RUNS = {
    # 每项可以是一个或多个 run 目录；多个会做逐步均值
    "eos-reward": ["/home/cxy/CodeGen-RLBench/final_tb_data/run_1_20250831_051403-eos-reward"],
    "dis-reward": ["/home/cxy/CodeGen-RLBench/final_tb_data/run_1_20250831_053524-dis-reward"],
    "eos-loss":   ["/home/cxy/CodeGen-RLBench/final_tb_data/run_1_20250831_055845-eos-loss"],
    "dis-loss":   ["/home/cxy/CodeGen-RLBench/final_tb_data/run_1_20250831_062128-dis-loss"],
}

# KL 的 scalar tag（若四个 run 的 tag 名相同，可以都填同一个）
TAGS = {
    "eos-reward": "Ppo/Mean_Kl",   # 替换成你面板里的完整标量路径
    "dis-reward": "Ppo/Mean_Kl",
    "eos-loss":   "Ppo/Mean_Kl",
    "dis-loss":   "Ppo/Mean_Kl",
}

# 平滑/坐标参数
EMA_ALPHA = 0.90
USE_SYMLOG = True          # True: symlog 纵轴，避免大数压扁小数
SYMLOG_LINTHRESH = 1e-3    # symlog 线性-对数切换阈值
ADD_INSET = False           # True: 右下角放大窗（看小范围）
INSET_YLIM = (0.0, 0.05)   # 放大窗 y 轴范围（按你的数据调整）
INSET_SIZE = ("42%", "42%")
INSET_LOC = 4              # 1 右上 / 2 左上 / 3 左下 / 4 右下
OUT_PREFIX = "ppo_kl_variants"
# ====== <<< 到此为止 >>> ======


def load_scalar_series(run_dir: str, tag: str):
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag '{tag}' not found in: {run_dir}")
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev], dtype=np.int64)
    vals = np.array([e.value for e in ev], dtype=np.float64)
    order = np.argsort(steps, kind="mergesort")
    steps, vals = steps[order], vals[order]
    uniq, idx = np.unique(steps, return_index=True)
    return uniq, vals[idx]


def aggregate_runs(run_dirs, tag):
    bucket = defaultdict(list)
    for rd in run_dirs:
        s, v = load_scalar_series(rd, tag)
        for step, val in zip(s, v):
            bucket[int(step)].append(float(val))
    if not bucket:
        raise RuntimeError(f"No data for tag '{tag}'")
    steps = np.array(sorted(bucket.keys()), dtype=np.int64)
    mean = np.array([np.mean(bucket[int(t)]) for t in steps], dtype=np.float64)
    return steps, mean


def ema(y, alpha=0.9):
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    if y.size == 0:
        return y
    m = y[0]
    out[0] = m
    for i in range(1, y.size):
        m = alpha * y[i] + (1 - alpha) * m
        out[i] = m
    return out


def plot_ppo_kl_variants(runs, tags, task_name,
                         ema_alpha=0.9, use_symlog=True, linthresh=1e-3,
                         add_inset=True, inset_ylim=(0.0, 0.05),
                         inset_size=("42%", "42%"), inset_loc=4,
                         out_prefix="ppo_kl_variants"):

    plt.figure(figsize=(6.2, 4.2))
    ax = plt.gca()

    # 保存每条曲线数据与颜色，给 inset 复用
    series, colors = {}, {}

    for label in ["eos-reward", "dis-reward", "eos-loss", "dis-loss"]:
        if label not in runs or label not in tags:
            continue
        steps, m = aggregate_runs(runs[label], tags[label])

        raw_line, = ax.plot(steps, m, linewidth=1.0, alpha=0.35, label=None)
        color = raw_line.get_color()
        colors[label] = color
        series[label] = (steps, m)

        ax.plot(steps, ema(m, alpha=ema_alpha), linewidth=2.0, label=label, color=color)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("KL to SFT (mean)")
    ax.set_xlim(left=0)

    if use_symlog:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    else:
        ax.set_ylim(bottom=0.0)

    ax.legend(loc="best", frameon=False, title="PPO variants")
    ax.set_title(f"PPO KL variants — {task_name.replace('_',' ')} (EMA α={ema_alpha:.2f})")
    plt.tight_layout()

    # 右下角放大窗（线性尺度，专看小范围，突出 eos/dis 间差异）
    if add_inset:
        axins = inset_axes(ax, width=inset_size[0], height=inset_size[1], loc=inset_loc, borderpad=0.8)
        for label, (steps, m) in series.items():
            axins.plot(steps, ema(m, alpha=ema_alpha), linewidth=1.6, color=colors[label])
        axins.set_xlim(left=0, right=max([steps.max() for steps, _ in series.values()]))
        axins.set_ylim(*inset_ylim)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("zoom", fontsize=8, pad=1.0)

    out = f"{out_prefix}_{task_name}"
    plt.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.png / {out}.pdf")


if __name__ == "__main__":
    plot_ppo_kl_variants(
        RUNS, TAGS, TASK_NAME,
        ema_alpha=EMA_ALPHA,
        use_symlog=USE_SYMLOG, linthresh=SYMLOG_LINTHRESH,
        add_inset=ADD_INSET, inset_ylim=INSET_YLIM,
        inset_size=INSET_SIZE, inset_loc=INSET_LOC,
        out_prefix=OUT_PREFIX,
    )

"""Plot the LoRA sweep: four graphs articulating the training-time / accuracy tradeoff.

Reads runs/<tag>.json (training: wall_s, history, args), runs/<tag>_eval.json
(motherduck parse/bind), runs/<tag>_bird.json (BIRD exec accuracy), plus ft_epoch1's
already-measured points. Writes PNGs into graphs/.

    python3 labs/plot_sweep.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
GRAPHS = HERE.parent / "graphs"

# tag -> (label, lora_r, lr, targets, color)
CONFIGS = {
    "sweep_r4":    ("r=4",    4,  1e-4, "all",    "#1f77b4"),
    "sweep_r8":    ("r=8",    8,  1e-4, "all",    "#2ca02c"),
    "ft_epoch1":   ("r=16",   16, 1e-4, "all",    "#9467bd"),
    "sweep_r32":   ("r=32",   32, 1e-4, "all",    "#ff7f0e"),
    "sweep_r64":   ("r=64",   64, 1e-4, "all",    "#d62728"),
    "sweep_lr3e5": ("lr 3e-5", 16, 3e-5, "all",    "#17becf"),
    "sweep_lr3e4": ("lr 3e-4", 16, 3e-4, "all",    "#e377c2"),
    "sweep_nomlp": ("mixers", 16, 1e-4, "mixers", "#8c564b"),
}

# ft_epoch1's eval numbers were measured as lab07_eval_ft_128_mn128 / lab10_ft_128.
EVAL_ALIAS = {"ft_epoch1": ("lab07_eval_ft_128_mn128", "lab10_ft_128")}

# Trainable params per config, from apply_lora's geometry (measured ft_epoch1 = 32.46M
# at r=16 all-targets): scale linearly in r; mixers excludes the MLP share (~56%).
PARAMS_M = {"sweep_r4": 8.1, "sweep_r8": 16.2, "ft_epoch1": 32.46,
            "sweep_r32": 64.9, "sweep_r64": 129.8,
            "sweep_lr3e5": 32.46, "sweep_lr3e4": 32.46, "sweep_nomlp": 14.4}

# ft_epoch1 trained on the OLD box (beta6 container): 116.6 min for the identical r=16
# recipe that takes ~81 min on this box's public stack. Marked differently in the graphs;
# do not read wall-time comparisons across that boundary.
OLD_STACK = {"ft_epoch1"}


def load(tag, suffix=""):
    for cand in (f"{tag}{suffix}.json",):
        p = RUNS / cand
        if p.exists():
            return json.loads(p.read_text())
    return None


def points():
    """One row per config: train + eval metrics."""
    rows = []
    for tag, (label, r, lr, targets, color) in CONFIGS.items():
        train = load(tag)
        ev_tag, bird_tag = EVAL_ALIAS.get(tag, (f"{tag}_eval", f"{tag}_bird"))
        ev, bird = load(ev_tag), load(bird_tag)
        if not train:
            print(f"  skip {tag}: no training json")
            continue
        rows.append({
            "tag": tag, "label": label, "r": r, "lr": lr, "targets": targets,
            "color": color, "params_m": PARAMS_M[tag],
            "wall_min": train["wall_s"] / 60,
            "tok_s": train["agg_tok_s"],
            "loss_first": train["loss_first_logged"],
            "loss_last": train["loss_last_logged"],
            "history": train.get("history", []),
            "parse": ev["parse_rate"] if ev else None,
            "bind": ev["bind_rate"] if ev else None,
            "exec": bird["exec_accuracy"] if bird else None,
        })
    return rows


def fig_pareto(rows):
    """Training wall time vs accuracy, both metrics, every config annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for ax, key, title in ((axes[0], "parse", "motherduck parse_rate"),
                           (axes[1], "exec", "BIRD exec_accuracy")):
        # Alternate annotation sides so nearby points don't overwrite each other.
        offsets = [(6, 4), (6, -12), (-46, 4), (-46, -12), (6, 10), (-52, 10)]
        plotted = [r for r in rows if r[key] is not None]
        plotted.sort(key=lambda r: r[key])
        for i, row in enumerate(plotted):
            old = row["tag"] in OLD_STACK
            ax.scatter(row["wall_min"], row[key], s=row["params_m"] * 3,
                       color=row["color"], alpha=0.45 if old else 0.85,
                       edgecolors="black", linewidths=0.5,
                       facecolors="none" if old else row["color"])
            ax.annotate(row["label"] + (" (old stack)" if old else ""),
                        (row["wall_min"], row[key]),
                        textcoords="offset points", xytext=offsets[i % len(offsets)],
                        fontsize=8)
        ax.set_xlabel("training wall time (min, 1 epoch)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("LoRA sweep: accuracy vs training time (marker size = trainable params)")
    fig.tight_layout()
    return fig


def fig_loss(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        h = row["history"]
        if not h:
            continue
        steps = [x["step"] for x in h]
        losses = [x["loss_mean"] for x in h]
        ax.plot(steps, losses, color=row["color"], label=row["label"], linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss by config")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_params(rows):
    """Parameter efficiency: trainable params (log) vs accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    rank_rows = [r for r in rows if r["lr"] == 1e-4 and r["parse"] is not None]
    for row in rank_rows:
        style = "s" if row["targets"] == "mixers" else "o"
        ax.scatter(row["params_m"], row["parse"], marker=style, s=70,
                   color=row["color"], edgecolors="black", linewidths=0.5,
                   label="mixers (no MLP)" if style == "s" else None)
        dy = -14 if row["label"] in ("mixers", "r=8") else 6
        ax.annotate(row["label"], (row["params_m"], row["parse"]),
                    textcoords="offset points", xytext=(6, dy), fontsize=8)
        if row["exec"] is not None:
            ax.scatter(row["params_m"], row["exec"], marker=style, s=70,
                       color=row["color"], alpha=0.45, edgecolors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (M, log)")
    ax.set_ylabel("accuracy (solid = parse, faded = exec)")
    ax.set_title("Parameter efficiency at 1 epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_lr(rows):
    lr_rows = sorted((r for r in rows if r["r"] == 16 and r["targets"] == "all"
                      and r["parse"] is not None), key=lambda r: r["lr"])
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(lr_rows))
    ax.plot(x, [r["parse"] for r in lr_rows], "o-", label="motherduck parse")
    ax.plot(x, [r["bind"] for r in lr_rows], "s--", label="motherduck bind")
    ax.plot(x, [r["exec"] for r in lr_rows], "^:", label="BIRD exec")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'{r["lr"]:.0e}' for r in lr_rows])
    ax.set_xlabel("learning rate (r=16, all targets)")
    ax.set_ylabel("accuracy")
    ax.set_title("Learning-rate sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    GRAPHS.mkdir(exist_ok=True)
    rows = points()
    if not rows:
        print("no sweep data yet")
        return 1
    for name, fn in (("pareto", fig_pareto), ("loss_curves", fig_loss),
                     ("param_efficiency", fig_params), ("lr_sensitivity", fig_lr)):
        fig = fn(rows)
        out = GRAPHS / f"{name}.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

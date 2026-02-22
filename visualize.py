"""
visualize.py — Research-Quality Figures
========================================
Generates all figures for the jailbreak evaluation report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

PALETTE = {
    "claude-3-opus":       "#1B4F72",
    "gpt-4":               "#1A5276",
    "llama-2-70b-chat":    "#E67E22",
    "mistral-7b-instruct": "#C0392B",
}

CATEGORY_LABELS = {
    "direct_request":       "Direct Request",
    "roleplay_persona":     "Roleplay / Persona",
    "hypothetical_framing": "Hypothetical Framing",
    "encoded_obfuscation":  "Encoded Obfuscation",
    "many_shot_jailbreak":  "Many-Shot Jailbreak",
    "authority_persona":    "Authority Persona",
    "competing_objectives": "Competing Objectives",
    "prompt_injection":     "Prompt Injection",
}

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

RESULTS_PATH = Path(__file__).parent / "results" / "eval_results.json"


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ── Figure 1: Overall Refusal Rate with 95% CI ───────────────────────────────

def fig_refusal_rate(data):
    models   = list(data["metrics"].keys())
    means    = [data["metrics"][m]["refusal_rate"]["mean"] for m in models]
    ci_lo    = [data["metrics"][m]["refusal_rate"]["ci_95"][0] for m in models]
    ci_hi    = [data["metrics"][m]["refusal_rate"]["ci_95"][1] for m in models]
    err_lo   = [means[i] - ci_lo[i] for i in range(len(models))]
    err_hi   = [ci_hi[i] - means[i] for i in range(len(models))]
    colors   = [PALETTE[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    bars = ax.bar(x, [m * 100 for m in means], color=colors, width=0.55,
                  alpha=0.88, zorder=3, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, [m * 100 for m in means],
                yerr=[[e * 100 for e in err_lo], [e * 100 for e in err_hi]],
                fmt="none", color="black", capsize=6, capthick=2, linewidth=2, zorder=4)

    # Annotate bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{mean:.0%}", ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=colors[i])

    model_labels = ["Claude 3 Opus", "GPT-4", "Llama 2 70B", "Mistral 7B"]
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_title("Figure 1. Overall Jailbreak Refusal Rate by Model (95% Bootstrap CI)", 
                 fontsize=13, fontweight="bold", pad=12)
    ax.axhline(y=100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_refusal_rate.png")
    plt.close()
    print("  [✓] Figure 1 saved")


# ── Figure 2: Heatmap — Refusal Rate by Model × Attack Category ──────────────

def fig_heatmap(data):
    models = list(data["metrics"].keys())
    categories = list(CATEGORY_LABELS.keys())

    matrix = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        for j, cat in enumerate(categories):
            cat_data = data["metrics"][model]["by_category"].get(cat)
            if cat_data:
                matrix[i, j] = cat_data["refusal_rate"]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories],
                       rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    model_labels = ["Claude 3 Opus", "GPT-4", "Llama 2 70B", "Mistral 7B"]
    ax.set_yticklabels(model_labels, fontsize=10)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.35 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Refusal Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ax.set_title("Figure 2. Refusal Rate Heatmap: Model × Attack Category",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_heatmap.png")
    plt.close()
    print("  [✓] Figure 2 saved")


# ── Figure 3: Severity Score Distribution ────────────────────────────────────

def fig_severity(data):
    models = list(data["metrics"].keys())
    model_labels = ["Claude 3 Opus", "GPT-4", "Llama 2 70B", "Mistral 7B"]

    # Reconstruct severity distributions from raw results
    severity_data = {}
    for model in models:
        raw = data["raw_results"][model]
        severity_data[model] = [r["severity_score"] for r in raw]

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = np.arange(len(models))
    vparts = ax.violinplot(
        [severity_data[m] for m in models],
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=False,
    )

    for i, (body, model) in enumerate(zip(vparts["bodies"], models)):
        body.set_facecolor(PALETTE[model])
        body.set_alpha(0.75)
        body.set_edgecolor("white")

    vparts["cmeans"].set_color("black")
    vparts["cmeans"].set_linewidth(2)
    for part in ["cbars", "cmins", "cmaxes"]:
        vparts[part].set_color("black")
        vparts[part].set_linewidth(1.2)

    # Overlay individual points
    for i, model in enumerate(models):
        sv = severity_data[model]
        jitter = np.random.uniform(-0.08, 0.08, len(sv))
        ax.scatter(i + jitter, sv, alpha=0.5, s=20, color=PALETTE[model],
                   edgecolors="white", linewidth=0.3, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Severity Score (0 = safe, 3 = highly harmful)", fontsize=11)
    ax.set_ylim(-0.2, 3.5)

    # Severity level annotations
    for level, label, colour in [(0.5, "Mild", "#F1C40F"), (1.5, "Moderate", "#E67E22"), (2.5, "Severe", "#C0392B")]:
        ax.axhline(level, color=colour, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(3.52, level, label, fontsize=9, va="center", color=colour)

    ax.set_title("Figure 3. Distribution of Harmful Output Severity Scores\n"
                 "(violin = distribution, dots = individual attacks, bar = mean)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_severity.png")
    plt.close()
    print("  [✓] Figure 3 saved")


# ── Figure 4: Transfer Matrix ─────────────────────────────────────────────────

def fig_transfer(data):
    models = list(data["metrics"].keys())
    model_labels = ["Claude 3\nOpus", "GPT-4", "Llama 2\n70B", "Mistral\n7B"]
    n = len(models)

    matrix = np.zeros((n, n))
    for i, src in enumerate(models):
        for j, tgt in enumerate(models):
            if src == tgt:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = data["transfer_matrix"].get(src, {}).get(tgt, 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # Diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, facecolor="#EAECEE", zorder=2))
        ax.text(i, i, "—", ha="center", va="center", fontsize=14, color="grey", zorder=3)

    # Annotate
    for i in range(n):
        for j in range(n):
            if i != j:
                val = matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color, zorder=4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_xlabel("Target Model", fontsize=11)
    ax.set_ylabel("Source Model (attacks that succeeded)", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("Transfer Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ax.set_title("Figure 4. Cross-Model Attack Transfer Rate\n"
                 "(% of attacks succeeding on source that also succeed on target)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_transfer.png")
    plt.close()
    print("  [✓] Figure 4 saved")


# ── Figure 5: Refusal Rate by Difficulty ─────────────────────────────────────

def fig_difficulty(data):
    models = list(data["metrics"].keys())
    model_labels = ["Claude 3 Opus", "GPT-4", "Llama 2 70B", "Mistral 7B"]
    difficulties = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(difficulties))
    width = 0.18

    for i, (model, label) in enumerate(zip(models, model_labels)):
        rates = []
        for d in difficulties:
            d_data = data["metrics"][model]["by_difficulty"].get(d)
            rates.append(d_data["refusal_rate"] * 100 if d_data else 0)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, rates, width=width * 0.9,
                      color=PALETTE[model], label=label, alpha=0.87,
                      edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(["Difficulty 1\n(Naive)", "Difficulty 2\n(Moderate)", "Difficulty 3\n(Sophisticated)"], fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.8)
    ax.set_title("Figure 5. Refusal Rate vs. Attack Sophistication by Model",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_difficulty.png")
    plt.close()
    print("  [✓] Figure 5 saved")


# ── Figure 6: Consistency Scores ─────────────────────────────────────────────

def fig_consistency(data):
    categories = list(CATEGORY_LABELS.keys())
    models = list(data["metrics"].keys())
    model_labels = ["Claude 3 Opus", "GPT-4", "Llama 2 70B", "Mistral 7B"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(categories))
    width = 0.18

    for i, (model, label) in enumerate(zip(models, model_labels)):
        consistencies = []
        for cat in categories:
            cat_data = data["metrics"][model]["by_category"].get(cat)
            consistencies.append(cat_data["consistency"] * 100 if cat_data else 100)
        offset = (i - 1.5) * width
        ax.bar(x + offset, consistencies, width=width * 0.9,
               color=PALETTE[model], label=label, alpha=0.87,
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c].replace(" ", "\n") for c in categories],
                       fontsize=9)
    ax.set_ylabel("Consistency (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.8)
    ax.set_title("Figure 6. Response Consistency by Attack Category\n"
                 "(100% = model always gives same class of response to variants within category)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_consistency.png")
    plt.close()
    print("  [✓] Figure 6 saved")


if __name__ == "__main__":
    print("Generating figures...")
    data = load_results()
    fig_refusal_rate(data)
    fig_heatmap(data)
    fig_severity(data)
    fig_transfer(data)
    fig_difficulty(data)
    fig_consistency(data)
    print(f"\nAll figures saved to {FIG_DIR}")

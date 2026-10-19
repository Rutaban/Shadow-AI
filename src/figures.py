"""All figures used in the paper."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import FIGURES, FLOWS, RESULTS

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
})

PALETTE = {
    "benign_chat": "#3B6FB6", "benign_docqa": "#6FA8DC",
    "exfil_code_chunked": "#C0392B", "exfil_doc_bulk": "#E67E22",
    "agentic_scrape": "#8E44AD", "exfil_obfuscated": "#7F1D1D",
}
NICE = {
    "benign_chat": "Benign chat", "benign_docqa": "Benign document question answering",
    "exfil_code_chunked": "Chunked code leak", "exfil_doc_bulk": "Bulk document leak",
    "agentic_scrape": "Agentic scraping", "exfil_obfuscated": "Obfuscated code leak",
}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print("figure", name)


# ------------------------------------------------------------------ fig 1
def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(-0.02, 5.0); ax.axis("off"); ax.grid(False)
    boxes = {
        "A": (0.10, 3.95, 2.25, 0.95, "Real corpora\ncode, business tables,\ninstruction prompts", "#DCE6F2"),
        "B": (2.75, 3.95, 2.10, 0.95, "Session scripts\nsix behaviours", "#EAD9F2"),
        "C": (5.25, 3.95, 2.10, 0.95, "TLS client and\ninference endpoint", "#EAD9F2"),
        "D": (7.75, 3.95, 2.15, 0.95, "Content matching\nbaseline on plaintext", "#E2EFDA"),
        "E": (5.25, 2.45, 2.10, 0.95, "Egress tap\nTLS record capture", "#FCE4D6"),
        "F": (0.10, 2.45, 2.25, 0.95, "UNSW-NB15\n257,673 labelled\nenterprise flows", "#DCE6F2"),
        "G": (2.15, 1.30, 5.20, 0.75, "Encrypted traffic metadata features", "#FCE4D6"),
        "H": (2.15, 0.05, 5.20, 0.92, "Behavioural models\nlogistic regression, random forest,\ngradient boosting, bidirectional LSTM", "#FFF2CC"),
        "I": (7.75, 0.55, 2.15, 1.60, "Evaluation\nF1, ROC AUC,\ngeneralisation,\nevasion, latency", "#FFF2CC"),
    }
    for x, y, w, h, text, colour in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=colour, edgecolor="#44546A", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8)
    arrows = [((2.35, 4.42), (2.75, 4.42)), ((4.85, 4.42), (5.25, 4.42)),
              ((7.35, 4.42), (7.75, 4.42)), ((6.30, 3.95), (6.30, 3.40)),
              ((6.30, 2.45), (6.30, 2.00)), ((4.75, 1.30), (4.75, 1.02)),
              ((7.35, 0.55), (7.75, 0.80)), ((8.82, 3.95), (8.82, 2.15)),
              ((1.22, 2.45), (1.22, 1.68)), ((1.22, 1.68), (2.15, 1.68))]
    for start_point, end_point in arrows:
        ax.add_patch(FancyArrowPatch(start_point, end_point, arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.8, color="#44546A"))
    _save(fig, "fig02_pipeline")


# ------------------------------------------------------------------ fig 2
def threat_model_diagram():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.8); ax.axis("off"); ax.grid(False)
    ax.add_patch(FancyBboxPatch((0.1, 0.75), 5.05, 3.85, boxstyle="round,pad=0.08",
                                facecolor="#F2F5FA", edgecolor="#8497B0", linewidth=0.9))
    ax.text(2.6, 4.32, "Enterprise perimeter", ha="center", fontsize=8.5, style="italic")
    ax.add_patch(FancyBboxPatch((6.05, 0.75), 3.85, 3.85, boxstyle="round,pad=0.08",
                                facecolor="#FBF3F3", edgecolor="#B08497", linewidth=0.9))
    ax.text(7.97, 4.32, "Third party inference providers", ha="center", fontsize=8.5,
            style="italic")
    inner = [(0.4, 2.85, 1.85, 0.9, "Source control\nand file shares", "#DCE6F2"),
             (2.75, 2.85, 2.1, 0.9, "Workstation\nbrowser session", "#DCE6F2"),
             (2.75, 1.25, 2.1, 0.9, "Unsanctioned\nagent runtime", "#DCE6F2"),
             (0.4, 1.25, 1.85, 0.9, "Internal service\nendpoints", "#DCE6F2")]
    for x, y, w, h, text, colour in inner:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=colour, edgecolor="#44546A", linewidth=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)
    ax.add_patch(FancyBboxPatch((5.3, 2.05), 0.72, 1.3, boxstyle="round,pad=0.05",
                                facecolor="#FCE4D6", edgecolor="#C55A11", linewidth=0.9))
    ax.text(5.66, 2.7, "Egress\ntap", ha="center", va="center", fontsize=7.2)
    for y, text in [(3.62, "api.openai.com"), (2.7, "api.anthropic.com"),
                    (1.62, "generativelanguage.\ngoogleapis.com")]:
        ax.add_patch(FancyBboxPatch((6.3, y - 0.36), 3.35, 0.74, boxstyle="round,pad=0.05",
                                    facecolor="#F4DDDD", edgecolor="#C0392B", linewidth=0.7))
        ax.text(7.97, y, text, ha="center", va="center", fontsize=7.2)
    arrows = [((2.25, 3.3), (2.75, 3.3)), ((2.25, 1.7), (2.75, 1.7)),
              ((4.85, 3.2), (5.3, 2.95)), ((4.85, 1.75), (5.3, 2.35)),
              ((6.02, 3.0), (6.3, 3.55)), ((6.02, 2.7), (6.3, 2.7)),
              ((6.02, 2.4), (6.3, 1.75))]
    for start_point, end_point in arrows:
        ax.add_patch(FancyArrowPatch(start_point, end_point, arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.8, color="#7F7F7F"))
    ax.text(5.0, 0.25, "TLS 1.3 throughout, no interception key material available "
                       "at the tap", ha="center", fontsize=7.6)
    _save(fig, "fig01_threat_model")


# ------------------------------------------------------------------ fig 3
def separation_figure(ai: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.1))
    for name, group in ai.groupby("scenario"):
        axes[0, 0].scatter(group["sbytes"] / 1024, group["upload_share"], s=4,
                           alpha=0.5, c=PALETTE[name], label=NICE[name], linewidths=0)
        axes[0, 1].scatter(group["dur"], group["srecords"], s=4, alpha=0.5,
                           c=PALETTE[name], linewidths=0)
        axes[0, 2].scatter(group["macro_gap_cv"], group["srec_mean"], s=4, alpha=0.5,
                           c=PALETTE[name], linewidths=0)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Uplink KiB"); axes[0, 0].set_ylabel("Uplink share")
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Duration (s)"); axes[0, 1].set_ylabel("Uplink records")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("Turn gap variation"); axes[0, 2].set_ylabel("Mean record size")

    fields = [("upload_share", "Uplink share", False),
              ("srec_mean", "Mean uplink record size", True),
              ("max_uplink_run", "Longest uplink run", False)]
    exfil = ai["label"] == 2
    for ax, (field, label, logx) in zip(axes[1], fields):
        benign = ai.loc[~exfil, field].replace(0, np.nan).dropna()
        leak = ai.loc[exfil, field].replace(0, np.nan).dropna()
        if logx:
            low = max(1e-3, min(benign.min(), leak.min()))
            bins = np.logspace(np.log10(low), np.log10(max(benign.max(), leak.max())), 32)
            ax.set_xscale("log")
        else:
            bins = np.linspace(0, max(benign.max(), leak.max()), 32)
        ax.hist(benign, bins=bins, alpha=0.65, color="#3B6FB6", density=True,
                label="Benign use")
        ax.hist(leak, bins=bins, alpha=0.65, color="#C0392B", density=True,
                label="Exfiltration")
        ax.set_xlabel(label); ax.set_ylabel("Density")
    axes[1, 0].legend(frameon=False, fontsize=6, loc="upper right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), markerscale=2.6, fontsize=6.5,
               columnspacing=1.4, handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "fig03_separation")


# ------------------------------------------------------------------ fig 5
def traces_figure():
    with open(FLOWS / "tls_record_traces.json") as handle:
        traces = json.load(handle)
    ai = pd.read_csv(FLOWS / "ai_endpoint_flows.csv")
    picks, shares = {}, {}
    for name in ["benign_chat", "exfil_code_chunked", "agentic_scrape"]:
        subset = ai[ai["scenario"] == name].sort_values("srecords", ascending=False)
        for _, row in subset.iterrows():
            trace = traces.get(str(row["flow_id"]))
            if trace and len(trace) > 120:
                picks[name] = trace
                shares[name] = row["upload_share"]
                break
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.1, 3.5))
    for ax, (name, trace) in zip(np.atleast_1d(axes), picks.items()):
        times = np.array([t for t, _, _ in trace])
        sizes = np.array([s for _, _, s in trace], dtype=float)
        dirs = np.array([d for _, d, _ in trace])
        signed = np.where(dirs == 0, sizes, -sizes) / 1024.0
        ax.vlines(times, 0, signed, linewidth=0.7,
                  color=np.where(dirs == 0, "#C0392B", "#3B6FB6"))
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("KiB", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{NICE[name]}, uplink share {shares[name]:.2f}",
                     loc="left", fontsize=8)
        span = max(abs(signed.max()), abs(signed.min()))
        ax.set_ylim(-span * 1.15, span * 1.15)
    axes[-1].set_xlabel("Time within session (s)", fontsize=7.5)
    handles = [Line2D([0], [0], color="#C0392B", linewidth=1.4, label="Uplink"),
               Line2D([0], [0], color="#3B6FB6", linewidth=1.4, label="Downlink")]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.5, ncol=2,
                   loc="upper right")
    fig.tight_layout()
    _save(fig, "fig04_record_traces")


# ------------------------------------------------------------------ fig 6
def roc_pr_figure(report_b: dict):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    colours = {"Logistic Regression": "#7F7F7F", "Random Forest": "#C0392B",
               "Gradient Boosting": "#E67E22", "Bidirectional LSTM": "#3B6FB6"}
    for name, curve in report_b["roc"].items():
        auc = report_b["holdout"][name]["roc_auc"]
        axes[0].plot(curve["fpr"], curve["tpr"], linewidth=1.2,
                     color=colours.get(name, "#333"), label=f"{name} ({auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", linewidth=0.7, color="#999")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC", fontsize=9)
    axes[0].legend(frameon=False, loc="lower right", fontsize=6)
    for name, curve in report_b["pr"].items():
        ap = report_b["holdout"][name]["pr_auc"]
        axes[1].plot(curve["recall"], curve["precision"], linewidth=1.2,
                     color=colours.get(name, "#333"), label=f"{name} ({ap:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision and recall", fontsize=9)
    axes[1].legend(frameon=False, loc="lower left", fontsize=6)
    history = pd.DataFrame(report_b["bilstm_history"])
    axes[2].plot(history["epoch"], history["train_loss"], color="#3B6FB6",
                 linewidth=1.1, label="Training")
    axes[2].plot(history["epoch"], history["val_loss"], color="#C0392B",
                 linewidth=1.1, label="Validation")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Cross entropy")
    axes[2].set_title("Recurrent model training", fontsize=9)
    axes[2].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig05_roc_pr_training")


# ------------------------------------------------------------------ fig 6
def confusion_and_importance(report_a: dict, report_b: dict, top: int = 12):
    fig = plt.figure(figsize=(7.1, 2.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.80, 1.35], wspace=1.15,
                            left=0.075, right=0.985, top=0.88, bottom=0.20)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    names_a = ["Benign", "Legacy", "Shadow AI"]
    cm_a = np.array(report_a["models"]["Gradient Boosting"]["confusion_matrix"], dtype=float)
    cm_b = np.array(report_b["holdout"]["Random Forest"]["confusion_matrix"], dtype=float)
    for ax, cm, names, title in [
            (axes[0], cm_a, names_a, "Three class task"),
            (axes[1], cm_b, ["Benign", "Exfiltration"], "AI bound task")]:
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.2, rotation=35,
                                                            ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
        ax.set_xlabel("Predicted", fontsize=7.5); ax.set_ylabel("Actual", fontsize=7.5)
        ax.set_title(title, fontsize=8.5); ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.1f}%", ha="center",
                        va="center", fontsize=5.8,
                        color="white" if norm[i, j] > 0.55 else "#222")
    pretty = {"srec_max": "max uplink record", "srec_p90": "uplink record p90",
              "sload": "uplink bytes", "srec_std": "uplink record sd",
              "srec_mean": "mean uplink record", "drec_std": "downlink record sd",
              "up_burst_max_bytes": "largest burst", "up_bytes_per_burst": "bytes per burst",
              "byte_asymmetry": "byte asymmetry", "upload_share": "uplink share",
              "sinpkt_p90": "uplink gap p90", "pkt_asymmetry": "packet asymmetry",
              "bytes_per_second_up": "uplink rate", "record_asymmetry": "record asymmetry",
              "dload": "downlink bytes", "drec_mean": "mean downlink record"}
    items = report_b["permutation_importance"][:top][::-1]
    y = np.arange(len(items))
    axes[2].barh(y, [i["mean"] for i in items], xerr=[i["std"] for i in items],
                 color="#3B6FB6", height=0.68, error_kw={"linewidth": 0.6, "ecolor": "#555"})
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([pretty.get(i["feature"], i["feature"].replace("_", " "))
                             for i in items], fontsize=6.2)
    axes[2].set_xlabel("Drop in F1 when permuted", fontsize=7.5)
    axes[2].set_title("Feature importance", fontsize=8.5)
    axes[2].tick_params(axis="x", labelsize=6.5)
    axes[2].yaxis.set_ticks_position("left")
    _save(fig, "fig06_confusion_importance")


# ------------------------------------------------------------------ fig 7
def outcomes_figure(report_c: dict, report_b: dict, report_e: dict, report_d: dict):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75))
    width = 0.34

    def annotate(ax, xs, values):
        for x, v in zip(xs, values):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=5.6, va="bottom")

    dlp = report_c["test_split"]["per_class"][1]
    ours = report_b["holdout"]["Random Forest"]["per_class"][1]
    x = np.arange(3)
    a_vals = [dlp["precision"], dlp["recall"], dlp["f1"]]
    b_vals = [ours["precision"], ours["recall"], ours["f1"]]
    axes[0].bar(x - width / 2, a_vals, width, color="#95A5A6", label="Content DLP")
    axes[0].bar(x + width / 2, b_vals, width, color="#C0392B", label="Behavioural")
    annotate(axes[0], x - width / 2, a_vals); annotate(axes[0], x + width / 2, b_vals)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Precision", "Recall", "F1"], fontsize=7)
    axes[0].set_title("Content matching vs behaviour", fontsize=8)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=1.0)

    order = ["clean_exfil", "obfuscated_payload", "timing_jitter", "chunk_morphing"]
    labels = ["None", "Payload\nobfusc.", "Timing\njitter", "Chunk\nmorphing"]
    present = [(l, report_e[k]) for l, k in zip(labels, order) if k in report_e]
    x = np.arange(len(present))
    rec = [v["recall"] for _, v in present]
    auc = [v["roc_auc"] for _, v in present]
    axes[1].bar(x - width / 2, rec, width, color="#C0392B", label="Recall")
    axes[1].bar(x + width / 2, auc, width, color="#3B6FB6", label="ROC AUC")
    annotate(axes[1], x - width / 2, rec); annotate(axes[1], x + width / 2, auc)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l for l, _ in present], fontsize=6.5)
    axes[1].set_title("Evasion applied", fontsize=8)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=1.0)

    short = {"exfil_code_chunked": "Chunked\ncode", "exfil_doc_bulk": "Bulk\ndocument",
             "agentic_scrape": "Agentic\nscraping"}
    keys = list(report_d.keys())
    x = np.arange(len(keys))
    rec = [report_d[k]["recall_on_held_out"] for k in keys]
    auc = [report_d[k]["roc_auc"] for k in keys]
    axes[2].bar(x - width / 2, rec, width, color="#C0392B", label="Recall")
    axes[2].bar(x + width / 2, auc, width, color="#3B6FB6", label="ROC AUC")
    annotate(axes[2], x - width / 2, rec); annotate(axes[2], x + width / 2, auc)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([short.get(k, k) for k in keys], fontsize=6.5)
    axes[2].set_title("Behaviour withheld from training", fontsize=8)
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=1.0)

    for ax in axes:
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Score", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    _save(fig, "fig07_outcomes")

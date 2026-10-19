"""Run every experiment and emit results plus figures."""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch

import experiments
import figures
from config import FIGURES, RESULTS


def main() -> None:
    started = time.time()
    ai, seq = experiments.load_ai()
    print(f"loaded {len(ai)} AI bound flows, sequences {seq.shape}")

    profile = experiments.scenario_profile(ai)
    profile.to_csv(RESULTS / "scenario_profile.csv", index=False)
    print(profile.to_string(index=False))

    print("\n[task A] three class task on the harmonised feature space")
    report_a = experiments.task_a(ai)
    print("\n[task B] AI bound binary task on the full metadata feature space")
    report_b = experiments.task_b(ai, seq)
    print("\n[task C] content matching DLP baseline")
    report_c = experiments.task_c_dlp(ai, report_b)
    print("\n[task D] leave one behaviour out generalisation")
    report_d = experiments.task_d_generalisation(ai)
    print("\n[task E] evasion robustness")
    report_e = experiments.task_e_evasion(ai)

    environment = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "runtime_seconds": round(time.time() - started, 1),
    }
    bundle = {"environment": environment, "task_a": report_a, "task_b": report_b,
              "task_c": report_c, "task_d": report_d, "task_e": report_e}
    with open(RESULTS / "results.json", "w") as handle:
        json.dump(bundle, handle, indent=1)

    figures.pipeline_diagram()
    figures.threat_model_diagram()
    figures.separation_figure(ai)
    figures.traces_figure()
    figures.roc_pr_figure(report_b)
    figures.confusion_and_importance(report_a, report_b)
    figures.outcomes_figure(report_c, report_b, report_e, report_d)

    print(f"\nfinished in {time.time() - started:.1f}s")
    print(json.dumps({k: report_b['holdout'][k]['macro_f1'] for k in report_b['holdout']},
                     indent=1))


if __name__ == "__main__":
    main()

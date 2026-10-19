"""Harmonise UNSW-NB15 flow records into the common feature space."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW

COMMON = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "smean", "dmean", "sinpkt", "dinpkt", "sjit", "djit",
    "upload_share", "byte_asymmetry", "pkt_asymmetry", "bytes_per_second_up",
]


def load_unsw() -> pd.DataFrame:
    parts = [
        pd.read_csv(RAW / "UNSW_hf_training-set.csv"),
        pd.read_parquet(RAW / "unsw_w_train.parquet"),
    ]
    frames = []
    for part in parts:
        part = part.copy()
        part.columns = [str(c).strip().lstrip("﻿") for c in part.columns]
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    return frame


def to_common(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["dur"] = frame["dur"].astype(float)
    out["spkts"] = frame["spkts"].astype(float)
    out["dpkts"] = frame["dpkts"].astype(float)
    out["sbytes"] = frame["sbytes"].astype(float)
    out["dbytes"] = frame["dbytes"].astype(float)
    out["rate"] = frame["rate"].astype(float)
    out["sload"] = frame["sload"].astype(float)
    out["dload"] = frame["dload"].astype(float)
    out["smean"] = frame["smean"].astype(float)
    out["dmean"] = frame["dmean"].astype(float)
    # UNSW-NB15 reports inter packet arrival and jitter in milliseconds.
    out["sinpkt"] = frame["sinpkt"].astype(float) / 1000.0
    out["dinpkt"] = frame["dinpkt"].astype(float) / 1000.0
    out["sjit"] = frame["sjit"].astype(float) / 1000.0
    out["djit"] = frame["djit"].astype(float) / 1000.0
    total = out["sbytes"] + out["dbytes"]
    out["upload_share"] = np.where(total > 0, out["sbytes"] / total, 0.0)
    out["byte_asymmetry"] = np.log1p(out["sbytes"]) - np.log1p(out["dbytes"])
    out["pkt_asymmetry"] = np.log1p(out["spkts"]) - np.log1p(out["dpkts"])
    out["bytes_per_second_up"] = np.where(out["dur"] > 0, out["sbytes"] / out["dur"], 0.0)
    return out[COMMON]


def ai_to_common(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload",
                 "dload", "smean", "dmean", "upload_share", "byte_asymmetry",
                 "pkt_asymmetry", "bytes_per_second_up"]:
        out[name] = frame[name].astype(float)
    out["sinpkt"] = frame["sinpkt_mean"].astype(float)
    out["dinpkt"] = frame["dinpkt_mean"].astype(float)
    out["sjit"] = frame["sjit"].astype(float)
    out["djit"] = frame["djit"].astype(float)
    return out[COMMON]

"""Flow features derived only from encrypted traffic metadata.

The inputs are the tuples produced by the tap: a wire timestamp, a direction
and a TLS record size. No plaintext is used anywhere in this module, which is
what allows the detector to run without decryption.
"""
from __future__ import annotations

import math

import numpy as np

from config import MSS
from tap import DOWN, UP

BURST_IDLE = 0.35  # seconds of quiet that terminates a burst


def _segments(record_bytes: int) -> int:
    """TLS records are mapped to segments at a fixed maximum segment size."""
    return max(1, math.ceil(record_bytes / MSS))


def _stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_max": 0.0,
                f"{prefix}_p90": 0.0, f"{prefix}_cv": 0.0}
    mean = float(values.mean())
    std = float(values.std())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_cv": float(std / mean) if mean > 0 else 0.0,
    }


def _discrete_entropy(sizes: np.ndarray) -> float:
    if sizes.size == 0:
        return 0.0
    bins = np.clip((sizes // 256).astype(int), 0, 128)
    counts = np.bincount(bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _bursts(times: np.ndarray, sizes: np.ndarray) -> tuple[int, float, float]:
    if times.size == 0:
        return 0, 0.0, 0.0
    order = np.argsort(times)
    t = times[order]
    s = sizes[order]
    boundaries = np.where(np.diff(t) > BURST_IDLE)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries, [t.size - 1]))
    volumes = np.array([s[a:b + 1].sum() for a, b in zip(starts, ends)], dtype=float)
    spans = t[ends] - t[starts]
    return int(starts.size), float(volumes.max()), float(spans.mean())


def flow_features(records: list[tuple[float, int, int]]) -> dict:
    if not records:
        return {}
    arr = np.array(records, dtype=float)
    times, directions, sizes = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
    duration = float(times.max() - times.min())

    up_mask = directions == UP
    down_mask = directions == DOWN
    up_t, up_s = times[up_mask], sizes[up_mask]
    down_t, down_s = times[down_mask], sizes[down_mask]

    sbytes = float(up_s.sum())
    dbytes = float(down_s.sum())
    spkts = int(sum(_segments(int(v)) for v in up_s))
    dpkts = int(sum(_segments(int(v)) for v in down_s))

    feats: dict[str, float] = {
        "dur": duration,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "spkts": float(spkts),
        "dpkts": float(dpkts),
        "srecords": float(up_s.size),
        "drecords": float(down_s.size),
        "smean": float(sbytes / spkts) if spkts else 0.0,
        "dmean": float(dbytes / dpkts) if dpkts else 0.0,
        "rate": float((spkts + dpkts) / duration) if duration > 0 else 0.0,
        "sload": float(sbytes * 8 / duration) if duration > 0 else 0.0,
        "dload": float(dbytes * 8 / duration) if duration > 0 else 0.0,
    }

    total = sbytes + dbytes
    feats["upload_share"] = float(sbytes / total) if total else 0.0
    feats["byte_asymmetry"] = float(math.log1p(sbytes) - math.log1p(dbytes))
    feats["pkt_asymmetry"] = float(math.log1p(spkts) - math.log1p(dpkts))
    feats["bytes_per_second_up"] = float(sbytes / duration) if duration > 0 else 0.0

    sinpkt = np.diff(np.sort(up_t)) if up_t.size > 1 else np.array([])
    dinpkt = np.diff(np.sort(down_t)) if down_t.size > 1 else np.array([])
    feats.update(_stats(sinpkt, "sinpkt"))
    feats.update(_stats(dinpkt, "dinpkt"))
    feats["sjit"] = float(np.abs(np.diff(sinpkt)).mean()) if sinpkt.size > 1 else 0.0
    feats["djit"] = float(np.abs(np.diff(dinpkt)).mean()) if dinpkt.size > 1 else 0.0

    feats.update(_stats(up_s, "srec"))
    feats.update(_stats(down_s, "drec"))
    feats["srec_entropy"] = _discrete_entropy(up_s)
    feats["drec_entropy"] = _discrete_entropy(down_s)
    feats["srec_distinct"] = float(np.unique((up_s // 256).astype(int)).size)
    feats["srec_full_share"] = float((up_s >= 16000).mean()) if up_s.size else 0.0

    bursts, largest, span = _bursts(up_t, up_s)
    feats["up_burst_count"] = float(bursts)
    feats["up_burst_max_bytes"] = largest
    feats["up_burst_mean_span"] = span
    feats["up_burst_rate"] = float(bursts / duration) if duration > 0 else 0.0
    feats["up_bytes_per_burst"] = float(sbytes / bursts) if bursts else 0.0

    # Rhythmicity of the uplink measured on the gaps that separate bursts.
    if up_t.size > 1:
        gaps = np.diff(np.sort(up_t))
        macro = gaps[gaps > BURST_IDLE]
        if macro.size > 1:
            feats["macro_gap_mean"] = float(macro.mean())
            feats["macro_gap_cv"] = float(macro.std() / macro.mean()) if macro.mean() else 0.0
        else:
            feats["macro_gap_mean"] = 0.0
            feats["macro_gap_cv"] = 0.0
    else:
        feats["macro_gap_mean"] = 0.0
        feats["macro_gap_cv"] = 0.0

    # Longest uninterrupted run of uplink records with no downlink in between.
    run = best = 0
    for direction in directions:
        if direction == UP:
            run += 1
            best = max(best, run)
        else:
            run = 0
    feats["max_uplink_run"] = float(best)
    feats["uplink_run_share"] = float(best / directions.size)
    return feats


def sequence_view(records: list[tuple[float, int, int]], length: int = 128) -> np.ndarray:
    """Signed record size and log gap sequence used by the recurrent model."""
    out = np.zeros((length, 3), dtype=np.float32)
    for index, (when, direction, size) in enumerate(records[:length]):
        previous = records[index - 1][0] if index else when
        out[index, 0] = math.log1p(size) * (1.0 if direction == UP else -1.0)
        out[index, 1] = math.log1p(max(0.0, when - previous))
        out[index, 2] = 1.0 if direction == UP else 0.0
    return out

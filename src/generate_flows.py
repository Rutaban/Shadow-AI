"""Drive every scenario across the tapped link and persist the measurements."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import FLOWS, SEED, SESSIONS, WORKERS
import client
import corpora
import dlp_baseline
import features
import llm_endpoint
import scenarios
from tap import EgressTap

_counter = threading.Lock()
_done = [0]


def _plaintext(session: dict) -> str:
    parts = []
    for step in session["steps"]:
        body = json.loads(step["body"].decode("utf-8"))
        for message in body["messages"]:
            if message["role"] == "user":
                parts.append(message["content"])
    return "\n".join(parts)


def _one(tap: EgressTap, token: int, name: str, seed: int, total: int) -> dict | None:
    rng = random.Random(seed)
    session = scenarios.BUILDERS[name](rng)
    text = _plaintext(session)
    summary = None
    for attempt in range(3):
        obs = tap.register(token)
        try:
            summary = client.run_session(tap.port, token, session)
            break
        except Exception as exc:  # a dropped session is retried, then reported
            if attempt == 2:
                print(f"  session {token} failed: {exc}", file=sys.stderr)
                return None
            time.sleep(0.4 * (attempt + 1))
    if summary is None:
        return None
    time.sleep(0.05)
    with obs.lock:
        records = sorted(obs.records)
    feats = features.flow_features(records)
    if not feats:
        return None
    row = {
        "flow_id": token,
        "scenario": session["scenario"],
        "label": session["label"],
        "ai_bound": session["ai_bound"],
        "obfuscation": session["obfuscation"],
        "evasion": session["evasion"],
        "n_requests": summary["requests"],
        "app_uploaded_bytes": summary["app_uploaded_bytes"],
        "app_downloaded_bytes": summary["app_downloaded_bytes"],
        "plaintext_chars": len(text),
    }
    row.update(feats)
    row.update({f"dlp_{k}": v for k, v in dlp_baseline.scan(text).items()})
    with _counter:
        _done[0] += 1
        if _done[0] % 50 == 0:
            print(f"  {_done[0]}/{total} sessions complete", flush=True)
    return {"row": row, "sequence": features.sequence_view(records),
            "records": [(round(t, 6), d, s) for t, d, s in records[:2000]]}


def main() -> None:
    corpora.code_files(); corpora.benign_prompts(); corpora.business_blocks()
    server, port = llm_endpoint.start_endpoint()
    tap = EgressTap(port, seed=SEED)
    print(f"endpoint on {port}, tap on {tap.port}")

    jobs = []
    token = 1
    for name, count in SESSIONS.items():
        for index in range(count):
            jobs.append((token, name, SEED + token))
            token += 1
    random.Random(SEED).shuffle(jobs)
    total = len(jobs)
    print(f"generating {total} sessions with {WORKERS} workers")

    started = time.time()
    rows, sequences, raw = [], [], {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_one, tap, tok, name, seed, total) for tok, name, seed in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            rows.append(result["row"])
            sequences.append(result["sequence"])
            raw[str(result["row"]["flow_id"])] = result["records"]

    frame = pd.DataFrame(rows).sort_values("flow_id").reset_index(drop=True)
    order = frame["flow_id"].tolist()
    index = {row["flow_id"]: seq for row, seq in zip(rows, sequences)}
    stacked = np.stack([index[i] for i in order])

    frame.to_csv(FLOWS / "ai_endpoint_flows.csv", index=False)
    np.save(FLOWS / "ai_endpoint_sequences.npy", stacked)
    with open(FLOWS / "tls_record_traces.json", "w") as handle:
        json.dump(raw, handle)
    print(f"wrote {len(frame)} flows in {time.time() - started:.1f}s")
    tap.close(); server.shutdown()


if __name__ == "__main__":
    main()

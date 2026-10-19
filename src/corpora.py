"""Loaders for the real corpora that supply payload content.

Nothing in this module fabricates text. Every byte that later travels across
the instrumented link originates from one of the public artefacts downloaded
into data/raw/corpora.
"""
from __future__ import annotations

import io
import json
import tarfile
from functools import lru_cache

import pandas as pd

from config import CORPORA


@lru_cache(maxsize=1)
def code_files() -> list[tuple[str, str]]:
    """Real source files taken from two widely used Python repositories."""
    out: list[tuple[str, str]] = []
    for archive in ("flask.tar.gz", "requests.tar.gz"):
        path = CORPORA / archive
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not member.name.endswith((".py", ".rst", ".cfg", ".toml", ".json")):
                    continue
                if member.size < 800 or member.size > 240_000:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                try:
                    text = handle.read().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                out.append((member.name, text))
    out.sort()
    return out


@lru_cache(maxsize=1)
def benign_prompts() -> list[str]:
    """Short natural language instructions from the public Alpaca corpus."""
    frame = pd.read_parquet(CORPORA / "alpaca.json")
    texts: list[str] = []
    for instruction, extra in zip(frame["instruction"], frame["input"]):
        prompt = str(instruction).strip()
        if extra and str(extra).strip():
            prompt = prompt + "\n" + str(extra).strip()
        if 20 <= len(prompt) <= 1200:
            texts.append(prompt)
    return texts


@lru_cache(maxsize=1)
def role_prompts() -> list[str]:
    frame = pd.read_csv(CORPORA / "awesome_prompts.csv")
    return [str(v).strip() for v in frame["prompt"] if isinstance(v, str) and len(v) > 40]


@lru_cache(maxsize=1)
def business_records() -> pd.DataFrame:
    """Real consumer complaint records used as structured business content."""
    frame = pd.read_csv(CORPORA / "biz_26k-consumer-complaints.csv", low_memory=False)
    return frame


@lru_cache(maxsize=1)
def business_blocks() -> list[str]:
    """Contiguous blocks of the real business table serialised as CSV text."""
    frame = business_records()
    blocks: list[str] = []
    step = 60
    header = ",".join(str(c) for c in frame.columns)
    for start in range(0, min(len(frame), 24_000) - step, step):
        chunk = frame.iloc[start:start + step]
        buf = io.StringIO()
        chunk.to_csv(buf, index=False, header=False)
        blocks.append(header + "\n" + buf.getvalue())
    return blocks


@lru_cache(maxsize=1)
def stock_table() -> str:
    return (CORPORA / "biz_2014_apple_stock.csv").read_text()


@lru_cache(maxsize=1)
def repo_manifest() -> list[dict]:
    """Structured inventory of the real repositories, used by the agent scenario."""
    entries = []
    for name, text in code_files():
        entries.append({
            "path": name,
            "bytes": len(text.encode("utf-8")),
            "lines": text.count("\n") + 1,
            "sha_prefix": f"{abs(hash(name)) % (16 ** 12):012x}",
        })
    return entries


def manifest_json(n: int, offset: int) -> str:
    entries = repo_manifest()
    window = [entries[(offset + i) % len(entries)] for i in range(n)]
    return json.dumps({"host": "build-node-01", "inventory": window}, indent=1)

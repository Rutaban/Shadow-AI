"""Session scripts for the six traffic behaviours studied in this work."""
from __future__ import annotations

import base64
import gzip
import json
import random

import corpora

MODELS = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]
PATHS = ["/v1/chat/completions", "/v1/messages"]


def _body(model: str, messages: list[dict], stream: bool) -> bytes:
    return json.dumps({
        "model": model,
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": messages,
    }).encode("utf-8")


def _human_gap(rng: random.Random) -> float:
    """Reading and typing pause between two turns of a real conversation."""
    return min(75.0, max(2.0, rng.lognormvariate(2.35, 0.75)))


def _machine_gap(rng: random.Random, mean: float, spread: float) -> float:
    return max(0.05, rng.gauss(mean, spread))


def _obfuscate(text: str, mode: str) -> str:
    raw = text.encode("utf-8")
    if mode == "base64":
        return base64.b64encode(raw).decode()
    if mode == "gzip_base64":
        return base64.b64encode(gzip.compress(raw)).decode()
    if mode == "hex":
        return raw.hex()
    if mode == "reversed_lines":
        return "\n".join(line[::-1] for line in text.split("\n"))
    return text


def benign_chat(rng: random.Random) -> dict:
    prompts = corpora.benign_prompts()
    turns = rng.randint(1, 8)
    history: list[dict] = []
    steps = []
    for index in range(turns):
        history = history + [{"role": "user", "content": prompts[rng.randrange(len(prompts))]}]
        steps.append({"body": _body(rng.choice(MODELS), history, True),
                      "gap": 0.0 if index == 0 else _human_gap(rng),
                      "stream": True})
        history = history + [{"role": "assistant", "content": ""}]
    return {"scenario": "benign_chat", "label": 0, "ai_bound": 1, "steps": steps,
            "obfuscation": "none", "evasion": "none"}


def benign_docqa(rng: random.Random) -> dict:
    files = corpora.code_files()
    blocks = corpora.business_blocks()
    prompts = corpora.benign_prompts()
    if rng.random() < 0.55:
        _, text = files[rng.randrange(len(files))]
    else:
        text = blocks[rng.randrange(len(blocks))]
    excerpt = text[: rng.randint(900, 5200)]
    question = prompts[rng.randrange(len(prompts))]
    history = [{"role": "user", "content": f"{question}\n\n{excerpt}"}]
    steps = [{"body": _body(rng.choice(MODELS), history, True), "gap": 0.0, "stream": True}]
    for _ in range(rng.randint(0, 3)):
        history = history + [{"role": "assistant", "content": ""},
                             {"role": "user", "content": prompts[rng.randrange(len(prompts))]}]
        steps.append({"body": _body(rng.choice(MODELS), history, True),
                      "gap": _human_gap(rng), "stream": True})
    return {"scenario": "benign_docqa", "label": 0, "ai_bound": 1, "steps": steps,
            "obfuscation": "none", "evasion": "none"}


def _code_payload(rng: random.Random) -> str:
    files = corpora.code_files()
    picked = [files[rng.randrange(len(files))] for _ in range(rng.randint(2, 6))]
    return "\n\n".join(f"# file: {name}\n{text}" for name, text in picked)


def exfil_code_chunked(rng: random.Random, obfuscation: str = "none",
                       evasion: str = "none") -> dict:
    payload = _obfuscate(_code_payload(rng), obfuscation)
    chunk = rng.randint(2200, 16000)
    pieces = [payload[i:i + chunk] for i in range(0, len(payload), chunk)][:40]
    history: list[dict] = []
    steps = []
    for index, piece in enumerate(pieces):
        if evasion == "morph":
            piece = piece[: max(600, int(len(piece) * rng.uniform(0.35, 1.0)))]
        instruction = "Continue reviewing this part of the repository." if index else \
            "Review this repository and explain the architecture."
        history = history + [{"role": "user", "content": f"{instruction}\n{piece}"}]
        if evasion == "jitter":
            gap = 0.0 if index == 0 else _human_gap(rng)
        else:
            gap = 0.0 if index == 0 else _machine_gap(rng, rng.uniform(0.5, 3.0), 0.35)
        steps.append({"body": _body(rng.choice(MODELS), history, True),
                      "gap": gap, "stream": True})
        history = history + [{"role": "assistant", "content": ""}]
    name = "exfil_obfuscated" if obfuscation != "none" else "exfil_code_chunked"
    return {"scenario": name, "label": 2, "ai_bound": 1, "steps": steps,
            "obfuscation": obfuscation, "evasion": evasion}


def exfil_doc_bulk(rng: random.Random) -> dict:
    blocks = corpora.business_blocks()
    steps = []
    history: list[dict] = []
    for index in range(rng.randint(1, 3)):
        count = rng.randint(6, 34)
        text = "\n".join(blocks[rng.randrange(len(blocks))] for _ in range(count))
        history = history + [{"role": "user",
                              "content": "Summarise the accounts in this export.\n" + text}]
        steps.append({"body": _body(rng.choice(MODELS), history, True),
                      "gap": 0.0 if index == 0 else _machine_gap(rng, rng.uniform(1.0, 6.0), 0.6),
                      "stream": True})
        history = history + [{"role": "assistant", "content": ""}]
    return {"scenario": "exfil_doc_bulk", "label": 2, "ai_bound": 1, "steps": steps,
            "obfuscation": "none", "evasion": "none"}


def agentic_scrape(rng: random.Random) -> dict:
    blocks = corpora.business_blocks()
    period = rng.uniform(0.45, 2.4)
    steps = []
    calls = rng.randint(12, 60)
    offset = rng.randrange(120)
    for index in range(calls):
        state = corpora.manifest_json(rng.randint(6, 26), offset + index * 7)
        table = blocks[(offset + index * 3) % len(blocks)][: rng.randint(1500, 9000)]
        content = f"Ingest the following host state and table slice.\n{state}\n{table}"
        steps.append({"body": _body(rng.choice(MODELS),
                                    [{"role": "user", "content": content}],
                                    rng.random() < 0.5),
                      "gap": 0.0 if index == 0 else _machine_gap(rng, period, period * 0.06),
                      "stream": True})
    return {"scenario": "agentic_scrape", "label": 2, "ai_bound": 1, "steps": steps,
            "obfuscation": "none", "evasion": "none"}


def exfil_obfuscated(rng: random.Random) -> dict:
    mode = rng.choice(["base64", "gzip_base64", "hex", "reversed_lines"])
    evasion = rng.choice(["none", "jitter", "morph"])
    return exfil_code_chunked(rng, obfuscation=mode, evasion=evasion)


BUILDERS = {
    "benign_chat": benign_chat,
    "benign_docqa": benign_docqa,
    "exfil_code_chunked": exfil_code_chunked,
    "exfil_doc_bulk": exfil_doc_bulk,
    "agentic_scrape": agentic_scrape,
    "exfil_obfuscated": exfil_obfuscated,
}

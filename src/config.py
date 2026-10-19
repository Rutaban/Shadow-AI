"""Central configuration for the Shadow AI exfiltration study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPORA = RAW / "corpora"
PROCESSED = DATA / "processed"
FLOWS = DATA / "flows"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (PROCESSED, FLOWS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260904

# Emulated wide area link used by the instrumented relay.
LINK_ONEWAY_MS_MEAN = 14.0
LINK_ONEWAY_MS_JITTER = 3.5
LINK_UPLINK_MBPS = 40.0
LINK_DOWNLINK_MBPS = 200.0

# Ethernet maximum segment size assumed when mapping TLS records to segments.
MSS = 1448
TLS_RECORD_MAX = 16384

# Number of AI endpoint sessions to generate per scenario.
SESSIONS = {
    "benign_chat": 620,
    "benign_docqa": 380,
    "exfil_code_chunked": 340,
    "exfil_doc_bulk": 300,
    "agentic_scrape": 300,
    "exfil_obfuscated": 240,
}
WORKERS = 48

LABELS = {"benign_enterprise": 0, "legacy_threat": 1, "shadow_ai_exfil": 2}

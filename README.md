# Characterizing Shadow AI Data Exfiltration Patterns in Enterprise Traffic

Reference implementation for the SOICT 2026 submission of the same name.

The project studies a question that content inspection cannot answer. When an
employee pastes a private repository or an internal customer table into an
unsanctioned assistant, the bytes leave the network inside an ordinary TLS
session to an ordinary inference endpoint. Nothing about the destination or the
protocol is anomalous, and the payload rarely carries the fixed identifiers a
commodity data loss prevention engine looks for. This code builds a measured
corpus of such sessions, extracts features that use only encrypted traffic
metadata, and shows that the behaviour separates cleanly from ordinary
assistant use.

## What the code does

1. Starts a local TLS endpoint that speaks the OpenAI and Anthropic chat
   request formats and answers with a token by token server sent event stream.
2. Routes every client session through a passive tap that parses TLS record
   headers off the ciphertext, timestamps each record, and forwards it over a
   shaped wide area link.
3. Drives six behaviours across that link using real content: source files from
   two public Python repositories, real consumer complaint records, and real
   instruction prompts.
4. Derives a flow feature vector from the record level observations alone. No
   plaintext reaches the detector.
5. Trains logistic regression, random forest, gradient boosting and a
   bidirectional LSTM, and compares them against a regular expression data loss
   prevention baseline that is given full plaintext access.
6. Measures generalisation to unseen exfiltration behaviours, robustness to
   payload obfuscation and timing evasion, and single flow inference latency.

## Layout

```
src/config.py          experiment constants
src/corpora.py         loaders for the real payload corpora
src/llm_endpoint.py    local TLS inference endpoint
src/tap.py             passive record level tap and link shaper
src/scenarios.py       the six session scripts
src/client.py          instrumented HTTPS client
src/features.py        encrypted traffic metadata features
src/dlp_baseline.py    regular expression content rules
src/unsw_loader.py     UNSW-NB15 harmonisation
src/models.py          tabular models and the Bi-LSTM
src/experiments.py     the five reported experiments
src/figures.py         every figure in the paper
src/generate_flows.py  traffic generation entry point
src/run_all.py         evaluation entry point
paper/                 the manuscript, its build script and the camera ready PDF
```

## Headline results

Measured corpus: 2,180 sessions, 1,000 benign and
1,180 exfiltration, 51 metadata features, plus
257,673 UNSW-NB15 records for the harmonised task. Every number below comes from
`results/results.json` and is reproduced by `src/run_all.py`.

| Task | Metric | Value |
| --- | --- | --- |
| Assistant bound binary, holdout | macro F1 (random forest) | 0.982 |
| Assistant bound binary, holdout | ROC AUC | 0.999 |
| Assistant bound binary, holdout | recall, precision | 0.977, 0.989 |
| Assistant bound binary, holdout | false positive rate | 1.33 percent |
| Assistant bound binary, holdout | macro F1 (Bi-LSTM, sequence only) | 0.888 |
| Content matching baseline, same holdout | recall with full plaintext | 0.017 |
| Transfer, behaviour never seen in training | recall range | 0.968 to 1.000 |
| Evasion, model trained only on non evasive sessions | recall, no evasion | 0.987 |
| Evasion | recall, payload obfuscation | 0.914 |
| Evasion | recall, timing jitter | 0.965 |
| Evasion | recall, chunk morphing | 0.786 |
| Deployment | single flow scoring latency, mean | 38.2 ms |

## Reproducing

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# follow data/raw/DOWNLOAD.md to fetch the public inputs
PYTHONPATH=src ./.venv/bin/python src/generate_flows.py
PYTHONPATH=src ./.venv/bin/python src/run_all.py
```

Traffic generation opens sockets on the loopback interface only and needs no
elevated privileges. It takes roughly two hours because session think times are
reproduced at their natural length rather than compressed.

Numeric results land in `results/results.json`, per scenario summaries in
`results/scenario_profile.csv`, and figures in `figures/`.

## Figures

| File | Appears as |
| --- | --- |
| `figures/fig01_threat_model.*` | Fig. 1, threat model |
| `figures/fig02_pipeline.*` | Fig. 2, measurement and evaluation pipeline |
| `figures/fig03_separation.*` | Fig. 3, session projections and marginal distributions |
| `figures/fig04_record_traces.*` | Fig. 4, TLS record traces |
| `figures/fig05_roc_pr_training.*` | Fig. 5, ROC, precision recall and training curves |
| `figures/fig06_confusion_importance.*` | Fig. 6, confusion matrices and feature importance |
| `figures/fig07_outcomes.*` | Fig. 7, baseline comparison, evasion and transfer |

## The paper

`paper/SOICT2026_Shadow_AI_Exfiltration_CCIS.pdf` is the camera ready manuscript
and `.doc` is the editable source. `paper/build_paper.py` rebuilds the manuscript
directly from `results/results.json`, so every number in the text and every
figure is generated from the measured corpus rather than typed by hand.

## Citing

Jania, R., Biswas, J., Ahmed, F.S., Hossain, M.F.: Characterizing shadow AI data
exfiltration patterns in enterprise traffic. SOICT 2026.

## Data provenance

All payload content is public. Source code payloads come from the Flask and
Requests repositories, business record payloads from a public consumer complaint
dataset, benign prompts from the Stanford Alpaca instruction corpus, and the
background enterprise traffic from UNSW-NB15. `data/raw/DOWNLOAD.md` lists every
source with its retrieval instructions. Nothing in this repository contains
private or personal data belonging to any organisation.

## Licence

MIT. See `LICENSE`.

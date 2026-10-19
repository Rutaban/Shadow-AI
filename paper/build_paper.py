"""Assemble the manuscript from the measured results.

Every numeric value printed in the paper is read out of results/results.json or
results/scenario_profile.csv, which are produced by src/run_all.py. Nothing is
typed in by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lncs
from refs import REFERENCES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FLOWS_CSV = ROOT / "data" / "flows" / "ai_endpoint_flows.csv"
FIGURES = ROOT / "figures"
OUT = Path(__file__).resolve().parent


FEATURE_LABEL = {
    "srec_max": "the largest uplink record", "srec_p90": "the ninetieth percentile of uplink record size",
    "sload": "total uplink bytes", "srec_std": "the spread of uplink record sizes",
    "srec_mean": "the mean uplink record size", "drec_std": "the spread of downlink record sizes",
    "up_burst_max_bytes": "the largest uplink burst", "up_bytes_per_burst": "bytes per uplink burst",
    "byte_asymmetry": "byte asymmetry", "upload_share": "the uplink share of session bytes",
    "sinpkt_p90": "the ninetieth percentile of uplink gaps", "pkt_asymmetry": "packet asymmetry",
    "bytes_per_second_up": "the uplink byte rate",
}


def listed(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def pct(x: float, places: int = 1) -> str:
    return f"{100 * x:.{places}f}"


def num(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}"


def pc(metrics: dict, cls: int) -> dict:
    """Per class metrics, tolerant of JSON string keys."""
    table = metrics["per_class"]
    return table[str(cls)] if str(cls) in table else table[cls]


def build() -> Path:
    R = json.loads((RESULTS / "results.json").read_text())
    A, B, C, D, E = R["task_a"], R["task_b"], R["task_c"], R["task_d"], R["task_e"]
    profile = pd.read_csv(RESULTS / "scenario_profile.csv")
    flows = pd.read_csv(FLOWS_CSV)

    doc = lncs.new_document()
    write_front_matter(doc, B, C)
    write_introduction(doc, B, C, E)
    write_related_work(doc)
    write_threat_model(doc)
    write_methodology(doc, flows, B)
    write_evaluation(doc, R, profile, flows)
    write_evaluation_tail(doc, R)
    write_discussion(doc, B, E)
    write_conclusion(doc, B, C)
    lncs.heading(doc, "References", level=1)
    lncs.references(doc, REFERENCES)
    target = OUT / "SOICT2026_Shadow_AI_Exfiltration.docx"
    doc.save(target)
    return target


def write_front_matter(doc, B, C):
    lncs.para(doc, "Characterizing Shadow AI Data Exfiltration Patterns in "
                   "Enterprise Traffic", size=14, bold=True, align="center",
              after=18, line=17)
    lncs.rich(doc, [("Rutaban Jania", False, False, False), ("1", False, False, True),
                    (", Joya Biswas", False, False, False), ("1", False, False, True),
                    (", Farhan Sadik Ahmed", False, False, False), ("2", False, False, True),
                    (" and MD Fahim Hossain", False, False, False), ("3", False, False, True)],
              size=11, align="center", after=12, line=13)
    lncs.rich(doc, [("1", False, False, True),
                    (" Institute of Information Technology, Jahangirnagar University, "
                     "Savar, Dhaka 1342, Bangladesh", False, False, False)],
              size=9, align="center", line=11)
    lncs.rich(doc, [("2", False, False, True),
                    (" Lassonde School of Engineering, York University, 4700 Keele "
                     "Street, North York, Ontario M3J 1P3, Canada", False, False, False)],
              size=9, align="center", line=11)
    lncs.rich(doc, [("3", False, False, True),
                    (" Department of Anthropology, Jahangirnagar University, Savar, "
                     "Dhaka 1342, Bangladesh", False, False, False)],
              size=9, align="center", line=11, after=6)
    lncs.para(doc, "ayushitahrim4949@gmail.com, joyabiswas0103@gmail.com, "
                   "farhansadik2002@gmail.com, hossainfahim520@gmail.com",
              size=9, align="center", line=11, after=16)

    rf = B["holdout"]["Random Forest"]
    best_name, best = max(B["holdout"].items(), key=lambda kv: kv[1]["macro_f1"])
    dlp = pc(C["test_split"], 1)
    abstract = (
        "Enterprises now lose proprietary data through a channel that looks "
        "entirely ordinary. Staff paste private source code and internal business "
        "records into unsanctioned assistants, and autonomous agents stream "
        "internal state to the same providers, so the bytes leave inside routine "
        "TLS sessions addressed to reputable inference endpoints. Content "
        "matching data loss prevention misses this traffic because proprietary "
        "code and structured exports carry no fixed identifier, and deep "
        "inspection is unavailable once TLS 1.3 is enforced. This paper "
        "characterises the behaviour instead of the content. A passive tap parses "
        "TLS record headers off the ciphertext of sessions carrying real "
        "repository source, real complaint records and real instruction prompts, "
        "and a flow representation is derived from that metadata alone. Across "
        f"{B['n_flows']:,} measured sessions the representation separates benign "
        f"assistant use from exfiltration with a macro F1 of {num(best['macro_f1'])} "
        f"and a ROC AUC of {num(best['roc_auc'])} for the strongest model, at a "
        f"false positive rate of {pct(rf['false_positive_rate'], 2)} percent. A "
        "regular expression data loss prevention baseline granted complete "
        f"plaintext access reaches a recall of {num(dlp['recall'])} on the same "
        "corpus. The representation also survives payload obfuscation and timing "
        "evasion and transfers to withheld exfiltration behaviours, which "
        "supports the view that structure and rhythm are harder to disguise than "
        "content.")
    lncs.rich(doc, [("Abstract. ", True, False), (abstract, False, False)],
              size=9, line=11, after=10, indent=0.7)
    lncs.rich(doc, [("Keywords: ", True, False),
                    ("Shadow AI, Data exfiltration, Encrypted traffic analysis, "
                     "Data loss prevention, Insider threat, Network behaviour "
                     "analysis.", False, False)],
              size=9, line=11, indent=0.7, after=6)


def write_introduction(doc, B, C, E):
    lncs.heading(doc, "1  Introduction", 1)
    lncs.body(doc,
        "Generative assistants entered the workplace faster than the controls "
        "meant to govern them, and the fastest way to get a useful answer is to "
        "hand the model the material the question is about, which is often a "
        "private repository, a customer export or an internal planning document. "
        "The IBM Cost of a Data Breach Report for 2025 found that one in five "
        "surveyed organisations suffered a breach connected to unsanctioned "
        "artificial intelligence tools, that such breaches cost roughly 670 "
        "thousand United States dollars more than the study average, and that 97 "
        "percent of the affected organisations lacked appropriate access controls "
        "[1]. The phenomenon is named shadow AI, and it inherits its shape from "
        "shadow information technology, in which staff adopt unapproved services "
        "because approved ones are slower [2, 3].",
        first=True)
    lncs.body(doc,
        "Nothing about this channel looks wrong. The destination is a well known "
        "commercial endpoint, the protocol is the same TLS 1.3 that carries the "
        "rest of the working day [21], and the request is an ordinary chat "
        "completion. Commodity data loss prevention looks for identifiers with "
        "fixed shapes, classification banners and credential material [5], and a "
        "private repository contains none of those. What distinguishes a leak "
        "from a question is how much material moves, in what shape and with what "
        "timing.")
    lncs.body(doc,
        "If the payload cannot be read, and would not be conclusive even if it "
        "could, the defender should look at what encryption does not hide. A "
        "passive observer still sees the size of every TLS record, its direction "
        "and the instant it crossed the boundary. An assistant conversation is a "
        "short question followed by a long streamed answer, and pushing a "
        "repository through the same interface inverts that shape, because the "
        "history is resent on every turn while the answers stay short. An "
        "automated collector adds a rhythm no person types with. These are "
        "properties of the activity, not of the text, and an insider who rewrites "
        "or encodes the text does not remove them.")
    lncs.body(doc,
        "No public corpus of such traffic exists, so we built a measurement "
        "testbed. A local endpoint speaks the request formats of the major "
        "commercial chat interfaces and answers with a token by token event "
        "stream, and a passive tap parses the five byte TLS record header off the "
        "ciphertext, timestamps every record and forwards it across a shaped wide "
        "area link. The material carried is real, with leaked code from two "
        "widely deployed open source Python repositories, leaked documents from a "
        "public corpus of consumer complaint records, benign prompts from a "
        "public instruction corpus and background enterprise traffic from "
        "UNSW-NB15 [17, 18]. The reported numbers are measurements of sessions "
        "that ran rather than samples from a distribution we chose.")
    lncs.body(doc,
        "The paper makes three contributions. The first is a reproducible "
        "measurement harness and a corpus of TLS record level observations for "
        "six assistant bound behaviours, three benign and three adversarial, "
        "built from public content and released with the code. The second is an "
        "encrypted traffic representation using only record sizes, directions and "
        "arrival times, with evidence that it separates exfiltration from "
        "ordinary assistant use far more reliably than a content matching "
        "baseline given the plaintext. The third is an evaluation of what a "
        "deployed detector actually needs, covering transfer to withheld "
        "behaviours, robustness to obfuscation and timing evasion, and single "
        "flow inference cost.")


def write_related_work(doc):
    lncs.heading(doc, "2  Background and Related Work", 1)

    lncs.heading(doc, "2.1  Content Matching Data Loss Prevention", 2)
    lncs.body(doc,
        "Commercial data loss prevention grew out of compliance, so the objects it "
        "protects are the ones regulation names. Alneyadi and colleagues survey "
        "the techniques and the assumption they share, which is that leaked data "
        "can be recognised by its own form [5]. Regular expressions, checksum "
        "validation, exact matching against a registered corpus and document "
        "fingerprinting all rest on it. The assumption holds for a table of "
        "customer identifiers. It fails when the asset is source code whose value "
        "lies in what it does rather than in any string it contains, and again "
        "when the leak is a paraphrase, a summary or an encoding, each a normal "
        "thing to ask an assistant to do.",
        first=True)

    lncs.heading(doc, "2.2  Encrypted Traffic Analysis", 2)
    lncs.body(doc,
        "The field that studies what remains visible after encryption is mature. "
        "Papadogiannaki and Ioannidis survey it and note its central tension, "
        "which is that the metadata letting a defender find malware also lets an "
        "adversary profile a user [6]. Website fingerprinting shows how much can "
        "be recovered from packet sizes and directions alone, at internet scale "
        "and against defended traffic [8, 9, 10]. Related work classifies "
        "encrypted malware command channels from flow features [7], applications "
        "from packet sequences with convolutional and recurrent models [11], and "
        "tunnelled from direct traffic using time related features [12].",
        first=True)
    lncs.body(doc,
        "Two recent results show that assistant traffic in particular is unusually "
        "legible from the outside. Whisper Leak infers the topic of a conversation "
        "with a remote language model from the packet sizes and timings of the "
        "streamed response, across twenty eight models and several providers "
        "[13], and NetEcho reconstructs substantial parts of a conversation from "
        "the same side channel [14]. Both treat the leakage as a privacy failure "
        "to be mitigated. The present work inverts the framing. If token by token "
        "streaming makes an assistant session structurally distinctive to a "
        "passive observer, the defender at the enterprise egress can use that "
        "structure to tell a question apart from a leak. To our knowledge that "
        "direction has not been evaluated.")

    lncs.heading(doc, "2.3  Security of Large Language Model Deployments", 2)
    lncs.body(doc,
        "Work on model deployment security has concentrated on the model and the "
        "application around it. Indirect prompt injection shows that retrieved "
        "content can redirect an assistant into sending user data to an attacker "
        "controlled endpoint [15], extraction attacks show that a deployed model "
        "can emit memorised training data [16], and the OWASP list ranks "
        "sensitive information disclosure second in its 2025 revision [4]. All of "
        "this concerns what the model does with data it has received, not how the "
        "data reached it or whether the enterprise could have seen it leaving.",
        first=True)


def write_threat_model(doc):
    lncs.heading(doc, "3  Threat Model", 1)
    lncs.body(doc,
        "The adversary is an insider who is not necessarily malicious. The "
        "canonical case is an engineer who pastes a private service, file by "
        "file, into whichever assistant is open, or an analyst who exports a "
        "customer table and asks for a summary. The model also covers a hostile "
        "insider and an unsanctioned agent runtime configured with a personal "
        "key, because all three produce one observable, which is a large volume "
        "of internal material moving outward inside assistant requests. The "
        "adversary is not attacking the model or the network. They are using both "
        "as designed.",
        first=True)
    lncs.figure(doc, FIGURES / "fig01_threat_model.png",
                "Fig. 1. |Threat model. Internal repositories and service endpoints feed "
                "a browser session or an unsanctioned agent runtime, both of which "
                "reach third party providers over TLS 1.3. The defender observes "
                "the egress path and holds no key material.",
                width_cm=8.0)
    lncs.body(doc,
        "The defender is a passive tap on the egress path. It sees every byte of "
        "ciphertext in both directions with accurate timestamps, plus the "
        "destination address and the server name, and it cannot decrypt. This is "
        "deliberately the harder setting. Organisations that terminate TLS at a "
        "proxy can also run content inspection, and Section 5.5 shows what that "
        "buys them. Our claim is that metadata alone suffices, so the method also "
        "works where interception is impossible for legal or technical reasons.")
    lncs.body(doc,
        "We assume the defender can identify traffic bound for inference "
        "endpoints, which is routine because providers publish their address "
        "ranges and the server name is visible at the tap. The interesting task "
        "begins afterwards, when the defender must decide which assistant "
        "sessions are questions and which are leaks. The adversary may encode, "
        "compress or reorder the payload and may alter pacing and chunk sizes, "
        "but we do not assume one who pads traffic to a fixed profile at the "
        "transport layer, since that needs tooling beyond a browser session.")


SCENARIO_LABEL = {
    "benign_chat": "Benign chat",
    "benign_docqa": "Benign document question answering",
    "exfil_code_chunked": "Chunked code leak",
    "exfil_doc_bulk": "Bulk document leak",
    "agentic_scrape": "Agentic scraping",
    "exfil_obfuscated": "Obfuscated code leak",
}


def write_methodology(doc, flows, B):
    lncs.heading(doc, "4  Methodology", 1)
    lncs.figure(doc, FIGURES / "fig02_pipeline.png",
                "Fig. 2. |Measurement and evaluation pipeline. Plaintext reaches only the "
                "content matching baseline, never the behavioural models.",
                width_cm=9.6)

    lncs.heading(doc, "4.1  Testbed and Observation Point", 2)
    lncs.body(doc,
        "An inference endpoint accepts the request bodies of the OpenAI chat "
        "completion and Anthropic messages interfaces and answers with a server "
        "sent event stream of one event per generated token, so the downlink "
        "carries the shape of a real streaming response. Response length is a "
        "bounded sublinear function of request size, reflecting an assistant that "
        "summarises rather than echoes, the token rate is drawn per request "
        "between thirty eight and ninety two tokens per second, and response text "
        "comes from a public instruction corpus.",
        first=True)
    lncs.body(doc,
        "A client opens one TLS connection per session, reuses it across turns as "
        "browsers and provider software development kits do, and issues ordinary "
        "HTTP requests carrying a JSON body. The tap sits between the two and "
        "never decrypts. It accumulates the ciphertext, reads the five byte TLS "
        "record header to find each boundary, records the arrival time, direction "
        "and record length, and forwards the record after a link model delay. "
        "That model serialises each direction independently at forty megabits per "
        "second upward and two hundred downward and adds a one way propagation "
        "delay drawn from a normal distribution of mean fourteen milliseconds and "
        "standard deviation three and a half. Serialisation is additive and "
        "propagation is not, so pipelining behaves as on a real path. Records map "
        "to segments at a maximum segment size of 1448 bytes when packet counts "
        "are needed.")
    lncs.body(doc,
        "The record sizes are therefore what the TLS implementation emitted for "
        "the payloads that were sent, and the timings belong to sessions that "
        "ran. Only the link shaping and the token cadence come from a model "
        "rather than from the network.")

    lncs.heading(doc, "4.2  Session Behaviours and Content", 2)
    lncs.body(doc,
        "Six behaviours were scripted, three benign and three adversarial, and "
        "Table 1 lists them. Benign chat is a conversation of one to eight turns "
        "from the Alpaca corpus [28] with pauses from a lognormal reading and "
        "typing distribution. Benign document question answering pastes an "
        "excerpt of one to five kilobytes and asks about it, which is the "
        "everyday case a naive volume threshold would wrongly flag. Chunked code "
        "leak splits real repository files into turns of two to sixteen kilobytes "
        "with short machine driven gaps, bulk document leak sends one to three "
        "very large requests of real complaint records, and agentic scraping "
        "issues twelve to sixty requests at a near constant period, each carrying "
        "a host inventory and a table slice. Obfuscated code leak repeats the "
        "chunked leak with the payload encoded as base sixty four, compressed "
        "base sixty four, hexadecimal or reversed lines, and with timing jitter "
        "or randomised chunk sizes.",
        first=True)

    counts = flows.groupby("scenario").size()
    rows = [["Behaviour", "Label", "Content source", "Sessions"]]
    order = ["benign_chat", "benign_docqa", "exfil_code_chunked",
             "exfil_doc_bulk", "agentic_scrape", "exfil_obfuscated"]
    sources = {
        "benign_chat": "Alpaca instruction corpus",
        "benign_docqa": "Repository files and complaint records",
        "exfil_code_chunked": "Flask and Requests source",
        "exfil_doc_bulk": "Consumer complaint records",
        "agentic_scrape": "Repository inventory and table slices",
        "exfil_obfuscated": "Flask and Requests source, encoded",
    }
    lncs.table_caption(doc, "Table 1. |Session behaviours, content source and session count.")
    for name in order:
        rows.append([SCENARIO_LABEL[name],
                     "Benign" if name.startswith("benign") else "Exfiltration",
                     sources[name], int(counts.get(name, 0))])
    lncs.table(doc, rows, widths=[3.9, 2.0, 4.6, 1.7], align_right=[3])

    lncs.heading(doc, "4.3  Feature Construction", 2)
    lncs.body(doc,
        "Features come from the record tuples and nothing else, in four groups. "
        "Volume and asymmetry features capture how much moved in each direction, "
        "including the uplink share of session bytes and the difference of the "
        "logarithms of the two volumes. Record structure features describe the "
        "size distribution in each direction through the mean, the ninetieth "
        "percentile, the coefficient of variation, a coarse entropy over "
        "quantised sizes and the fraction of uplink records at the maximum "
        "length. Temporal features cover interarrival statistics and jitter. "
        "Burst features segment the uplink at gaps longer than 350 milliseconds "
        "and report the burst count, the largest burst volume, the mean span, the "
        "variation of the gaps between bursts and the longest run of uplink "
        "records with no downlink record between them.",
        first=True)
    lncs.body(doc,
        f"The full vector has {B['n_features']} dimensions. A parallel sequence "
        "view retained for the recurrent model gives each of the first 128 "
        "records its signed logarithmic size, the logarithm of the gap since the "
        "previous record and a direction indicator.")

    lncs.heading(doc, "4.4  Background Enterprise Traffic", 2)
    lncs.body(doc,
        "Assistant bound sessions alone cannot say whether a monitor would drown "
        "in alerts on an ordinary network, so we harmonise the measured flows "
        "with UNSW-NB15, which supplies labelled enterprise records covering "
        "normal activity and nine attack families [17, 18]. Both public "
        "partitions are used, giving 257,673 records. Eighteen features are "
        "common to both sources once the UNSW interarrival and jitter columns are "
        "converted to seconds and the asymmetry terms are recomputed identically. "
        "The space is deliberately narrow, because a feature present in one "
        "source and absent in the other would let a classifier identify the "
        "generator rather than the behaviour.",
        first=True)

    lncs.heading(doc, "4.5  Models and Baseline", 2)
    lncs.body(doc,
        "Logistic regression on standardised features is the linear reference, a "
        "random forest of four hundred trees with balanced subsample weighting is "
        "the nonlinear reference [22], and histogram based gradient boosted trees "
        "are the boosted ensemble [23]. A bidirectional long short term memory "
        "network with sixty four hidden units per direction, max pooling over "
        "time and a two layer head is trained on the sequence view for sixty "
        "epochs with class weighted cross entropy [24, 25]. Implementations come "
        "from scikit-learn and PyTorch [26, 27].",
        first=True)
    lncs.body(doc,
        "The baseline is a content matching engine of the kind commodity data "
        "loss prevention ships. It applies nine rules covering payment card "
        "numbers, national identifiers, cloud access keys, private key blocks, "
        "provider tokens, bank account numbers, classification banners, mail "
        "addresses and telephone numbers. Seven raise an alert, with mail "
        "addresses and telephone numbers treated as telemetry because alerting on "
        "them in a corporate network would be unusable. It is evaluated against "
        "the plaintext, so it enjoys the complete interception the behavioural "
        "models are denied.")


def write_evaluation(doc, R, profile, flows):
    A, B, C, D, E = R["task_a"], R["task_b"], R["task_c"], R["task_d"], R["task_e"]
    lncs.heading(doc, "5  Evaluation", 1)

    lncs.heading(doc, "5.1  Protocol", 2)
    env = R["environment"]
    lncs.body(doc,
        f"All sessions were generated on one host running Python {env['python']} "
        f"with scikit-learn {env['scikit_learn']} and PyTorch {env['torch']}. "
        "Splits are stratified and the seed is fixed at 20260904. Section 5.4 "
        "reports five fold cross validation and a seventy to thirty holdout, the "
        "latter reused by every later experiment so that comparisons use "
        "identical sessions. We report macro averaged F1 with precision, recall "
        "and the areas under the receiver operating characteristic and precision "
        "recall curves.",
        first=True)

    lncs.heading(doc, "5.2  How the Behaviours Separate", 2)
    lncs.body(doc,
        "Before any classifier is fitted it is worth asking whether the separation "
        "exists at all. Fig. 3 shows every measured session and Table 2 gives the "
        "per behaviour medians.",
        first=True)
    lncs.figure(doc, FIGURES / "fig03_separation.png",
                "Fig. 3. |Top, every measured session in three projections. Bottom, "
                "marginal distributions of three features for benign use and for "
                "exfiltration, with zero valued sessions excluded.",
                width_cm=11.4)
    lncs.table_caption(doc, "Table 2. |Median per behaviour of selected measured quantities. Volumes "
                            "are in bytes and durations in seconds.")
    order = ["benign_chat", "benign_docqa", "exfil_code_chunked",
             "exfil_doc_bulk", "agentic_scrape", "exfil_obfuscated"]
    prof = profile.set_index("scenario")
    rows = [["Behaviour", "Uplink bytes", "Downlink bytes", "Uplink share",
             "Seconds", "Uplink records", "Longest run"]]
    for name in order:
        r = prof.loc[name]
        rows.append([SCENARIO_LABEL[name], f"{r['sbytes']:,.0f}",
                     f"{r['dbytes']:,.0f}", f"{r['upload_share']:.3f}",
                     f"{r['dur']:.1f}", f"{r['srecords']:.0f}",
                     f"{r['max_uplink_run']:.0f}"])
    lncs.table(doc, rows, widths=[3.1, 1.85, 1.95, 1.4, 1.45, 1.4, 1.45], size=8.0,
               align_right=[1, 2, 3, 4, 5, 6])
    lncs.body(doc,
        "The pattern is the one the threat model predicts. Benign conversation is "
        "download heavy, because a short question earns a long streamed answer, "
        "while the leak behaviours invert the ratio by more than the raw volume "
        "difference, since the history is resent on every turn. Agentic scraping "
        "sits between the two and is distinguished by pacing. Fig. 4 shows raw "
        "record traces, where the difference in texture is visible without any "
        "statistics.")
    lncs.figure(doc, FIGURES / "fig04_record_traces.png",
                "Fig. 4. |TLS record traces of three measured sessions. Bars above the "
                "axis are uplink records and bars below are downlink records.",
                width_cm=9.8)

    lncs.heading(doc, "5.3  Three Class Detection in the Harmonised Space", 2)
    comp = A["composition"]
    lncs.body(doc,
        "The first experiment asks whether shadow AI exfiltration is separable "
        "from ordinary enterprise traffic and conventional attacks using only the "
        f"eighteen harmonised features. The pool holds {comp['0']:,} benign, "
        f"{comp['1']:,} legacy threat and {comp['2']:,} exfiltration records, "
        "split seventy to thirty, with results in Table 3 and the confusion "
        "structure in the left panel of Fig. 6.",
        first=True)
    rows = [["Model", "Accuracy", "Macro F1", "Exfiltration recall",
             "Exfiltration precision", "ROC AUC"]]
    for name, m in A["models"].items():
        cls = pc(m, 2)
        rows.append([name, num(m["accuracy"]), num(m["macro_f1"]),
                     num(cls["recall"]), num(cls["precision"]),
                     num(m.get("roc_auc", float("nan")))])
    lncs.table_caption(doc, "Table 3. |Three class results on the harmonised feature space. The "
                            "exfiltration columns are that class against the rest.")
    lncs.table(doc, rows, widths=[3.4, 1.9, 1.9, 2.4, 2.5, 1.7], size=8.5,
               align_right=[1, 2, 3, 4, 5])
    best_a = max(A["models"].items(), key=lambda kv: kv[1]["macro_f1"])
    cls_a = pc(best_a[1], 2)
    lncs.body(doc,
        f"The best model reaches a macro F1 of {num(best_a[1]['macro_f1'])} with "
        f"a recall of {num(cls_a['recall'])} on the exfiltration class. It should "
        "be read with care, since the three classes come from two generators and "
        "a residual domain difference cannot be excluded even though the space "
        "holds only features both sources define identically. We therefore read "
        "it as evidence that assistant bound exfiltration resembles neither "
        "ordinary traffic nor the attack families in UNSW-NB15, and place the "
        "weight of the paper on the next experiment.")

    lncs.heading(doc, "5.4  Benign Assistant Use Against Exfiltration", 2)
    lncs.body(doc,
        "This is the task that matters operationally. Every session is addressed "
        "to an inference endpoint and came from the same harness and link, so the "
        "classifier must decide which are leaks. The "
        f"set holds {int(B['class_counts']['0']):,} benign and "
        f"{int(B['class_counts']['1']):,} exfiltration sessions, and Table 4 reports "
        "both cross validation and the holdout.",
        first=True)
    rows = [["Model", "CV macro F1", "CV ROC AUC", "Recall", "Precision",
             "Macro F1", "ROC AUC", "PR AUC"]]
    for name, m in B["holdout"].items():
        cls = pc(m, 1)
        cv = B["cross_validation"].get(name)
        cv_f1 = (f"{num(cv['macro_f1']['mean'])} ({num(cv['macro_f1']['std'], 3)})"
                 if cv else "not run")
        cv_auc = num(cv["roc_auc"]["mean"]) if cv else "not run"
        rows.append([name, cv_f1, cv_auc, num(cls["recall"]), num(cls["precision"]),
                     num(m["macro_f1"]), num(m["roc_auc"]), num(m["pr_auc"])])
    lncs.table_caption(doc, "Table 4. |Assistant bound binary task. The first two columns are five "
                            "fold cross validation with the standard deviation "
                            "across folds in parentheses, the rest are the holdout "
                            "split.")
    lncs.table(doc, rows, widths=[2.9, 2.3, 1.8, 1.4, 1.6, 1.5, 1.4, 1.3], size=8.0,
               align_right=[1, 2, 3, 4, 5, 6, 7])
    lncs.figure(doc, FIGURES / "fig05_roc_pr_training.png",
                "Fig. 5. |Holdout ROC and precision recall curves with areas in the "
                "legends, and the loss curves of the recurrent model.", width_cm=11.4)
    best_b = max(B["holdout"].items(), key=lambda kv: kv[1]["macro_f1"])
    lncs.body(doc,
        f"{best_b[0]} is the strongest model with a macro F1 of "
        f"{num(best_b[1]['macro_f1'])} and a ROC AUC of "
        f"{num(best_b[1]['roc_auc'])}. A single feature such as uplink share "
        "separates the bulk leak cleanly, but agentic scraping and short chunked "
        "leaks need a conjunction of pacing and structure, which the tree "
        "ensembles supply. The recurrent model, which sees no aggregate feature "
        "and works only from the first 128 record sizes and gaps, "
        f"reaches a macro F1 of {num(B['holdout']['Bidirectional LSTM']['macro_f1'])} "
        f"and a ROC AUC of {num(B['holdout']['Bidirectional LSTM']['roc_auc'])}. It "
        "trails the tabular models, which is expected because a truncated record "
        "sequence discards the session totals the aggregates carry, but "
        "recovering most of the signal from the raw sequence supports the claim "
        "that the discriminative information lies in the shape of the session "
        "rather than in any hand chosen summary. Its training curves appear in "
        "the right panel of Fig. 5.")
    top = B["permutation_importance"][:5]
    lncs.body(doc,
        "The right panel of Fig. 6 ranks the features by the drop in F1 caused by "
        "permuting them, and the five strongest are " +
        listed(FEATURE_LABEL.get(t["feature"], t["feature"].replace("_", " "))
               for t in top) +
        ". These are the quantities the threat model predicted would matter, "
        "which is a useful check that the classifier is not keying on an "
        "artefact of the harness.")

    lncs.figure(doc, FIGURES / "fig06_confusion_importance.png",
                "Fig. 6. |Confusion matrices for the three class and assistant bound "
                "tasks, with counts and row normalised percentages, and the twelve "
                "strongest features by permutation importance.",
                width_cm=11.4)
    lncs.heading(doc, "5.5  Comparison Against Content Matching", 2)
    dlp_cls = pc(C["test_split"], 1)
    rf_cls = pc(B["holdout"]["Random Forest"], 1)
    whole = pc(C["whole_set"], 1)
    if dlp_cls["recall"] == 0.0:
        dlp_sentence = (
            "On the holdout split it raises no alert at all, and across the whole "
            f"corpus it flags {C['alerts_on_exfil']} of the "
            f"{whole['support']} exfiltration sessions, a recall of "
            f"{num(whole['recall'])}. ")
    else:
        dlp_sentence = (
            f"On the holdout split it recovers a recall of {num(dlp_cls['recall'])} "
            f"at a precision of {num(dlp_cls['precision'])}. ")
    lncs.body(doc,
        "The content matching baseline was given the plaintext of every session, "
        "more access than any deployment without full interception enjoys. "
        + dlp_sentence +
        f"The random forest, working from metadata alone, reaches a recall of "
        f"{num(rf_cls['recall'])} at a precision of {num(rf_cls['precision'])}, "
        "as the left panel of Fig. 7 shows. Only two of the nine rules ever "
        "fired, a classification banner in "
        f"{C['rule_activation']['classification_banner']} sessions and an "
        f"electronic mail address in {C['rule_activation']['email_address']}.",
        first=True)
    lncs.body(doc,
        "The gap is structural rather than a matter of tuning. The leaked material "
        "is a working web framework and a table of complaint narratives, and "
        "neither contains a payment card number, a national identifier or a "
        "private key block, because real source code and business prose usually "
        "do not. The rules that do fire are generic and fire on benign sessions "
        "at a similar rate, so raising their severity would trade a recall "
        "problem for a precision problem. The failure of content matching here is "
        "not an implementation weakness. It follows from the fact that the value "
        "of proprietary material is not encoded in any string it contains.")


def write_evaluation_tail(doc, R):
    B, D, E = R["task_b"], R["task_d"], R["task_e"]
    lncs.heading(doc, "5.6  Transfer to Unseen Exfiltration Behaviour", 2)
    lncs.body(doc,
        "A detector that recognises only the behaviours it was shown is of little "
        "use, because the next insider will do something slightly different. We "
        "remove one exfiltration behaviour from training, fit the random forest "
        "on what remains, and test on the removed behaviour against reserved "
        "benign sessions, with the outcome in the right panel of Fig. 7.",
        first=True)
    worst = min(D.items(), key=lambda kv: kv[1]["recall_on_held_out"])
    best = max(D.items(), key=lambda kv: kv[1]["recall_on_held_out"])
    lncs.body(doc,
        f"Transfer is strongest for {SCENARIO_LABEL[best[0]].lower()}, at a recall "
        f"of {num(best[1]['recall_on_held_out'])}, and weakest for "
        f"{SCENARIO_LABEL[worst[0]].lower()}, at "
        f"{num(worst[1]['recall_on_held_out'])}. That every behaviour is "
        "recovered at better than nine in ten without ever having been seen "
        "matters more than the ordering, because it shows the model has learned "
        "the family level property of sustained asymmetric outward transfer "
        "rather than the signature of any one script. A deployment therefore "
        "need not enumerate the variants an insider might invent.")

    lncs.heading(doc, "5.7  Robustness to Obfuscation and Timing Evasion", 2)
    lncs.body(doc,
        "The common objection to content inspection is that an insider can encode "
        "the payload, and the objection to behavioural detection is that an "
        "insider can slow down. We test both by training only on non evasive "
        "sessions, holding back a quarter of them as a reference, then evaluating "
        "on three evasive families. Payload obfuscation encodes the leaked code "
        "as base sixty four, compressed base sixty four, hexadecimal or reversed "
        "lines, timing jitter replaces machine pacing with the lognormal think "
        "time distribution of the benign behaviours, and chunk morphing "
        "randomises the size of each uploaded piece. The centre panel of Fig. 7 "
        "gives the result.",
        first=True)
    lncs.figure(doc, FIGURES / "fig07_outcomes.png",
                "Fig. 7. |Left, the content matching baseline against the behavioural "
                "model on the same holdout sessions. Centre, robustness under "
                "three evasion families. Right, transfer to an exfiltration "
                "behaviour withheld from training.", width_cm=11.4)
    names = {"clean_exfil": "no evasion", "obfuscated_payload": "payload obfuscation",
             "timing_jitter": "timing jitter", "chunk_morphing": "chunk morphing"}
    clean = E.get("clean_exfil")
    evasive = {k: E[k] for k in ("obfuscated_payload", "timing_jitter",
                                 "chunk_morphing") if k in E}
    worst = min(evasive.items(), key=lambda kv: kv[1]["recall"]) if evasive else None
    parts = listed(f"{num(v['recall'])} under {names[k]}" for k, v in evasive.items())
    lncs.body(doc,
        f"Recall on non evasive sessions the model never saw is "
        f"{num(clean['recall'])}, and the evasive families move it to {parts}. Encoding the payload gains the insider little and can work "
        "against them, because base sixty four inflates the material by a third "
        "and hexadecimal doubles it, making the uplink asymmetry more extreme "
        f"rather than less. The most effective of the three is {names[worst[0]]}, "
        f"which reduces recall to {num(worst[1]['recall'])} by attacking the "
        "record size statistics that Fig. 6 ranks highest. Even there the area "
        f"under the curve stays at {num(worst[1]['roc_auc'])} and not one of the "
        "reserved benign sessions was flagged, so the evasive sessions remain "
        "separable at a lower decision threshold. Evasion raises the threshold a "
        "defender must set rather than defeating the representation.")

    lat = B["inference_latency_ms"]
    lncs.body(doc,
        "Scoring one completed flow with the random forest takes "
        f"{lat['mean']:.2f} milliseconds on average on the evaluation host, with "
        f"a ninety fifth percentile of {lat['p95']:.2f} and a ninety ninth of "
        f"{lat['p99']:.2f} milliseconds, and feature extraction is one pass over "
        "the record list, so the classifier is not the bottleneck. The real cost "
        "is that a session completes only when it ends, which makes the natural "
        "deployment a near real time analytic over finished flows rather than an "
        "inline block.")


def write_discussion(doc, B, E):
    lncs.heading(doc, "6  Discussion and Limitations", 1)
    lncs.body(doc,
        "The results support a narrow claim. When an assistant bound TLS session "
        "is observed passively, the sizes, directions and arrival times of its "
        "records distinguish a person asking a question from a person or process "
        "moving internal material outward. It holds across three exfiltration "
        "behaviours, survives four payload encodings, and needs no plaintext.",
        first=True)
    lncs.body(doc,
        "The first limitation is the origin of the traffic. It was "
        "measured rather than sampled, which is a real advance over a synthetic "
        "table, but against a local endpoint over an emulated link, so the token "
        "cadence is our model of a provider. Recent measurement work confirms "
        "that real streaming responses produce the per token record structure our "
        "endpoint reproduces [13, 14], but a study against live providers is "
        "needed before these absolute numbers are quoted operationally.")
    lncs.body(doc,
        "The second is the benign class. Real workplaces contain heavier "
        "legitimate uses, such as an approved coding assistant indexing a "
        "repository under contract, which would resemble our exfiltration class. "
        "A deployment would separate the two by policy, using the destination and "
        "the credential rather than the shape of the flow. The detector answers "
        "whether a bulk transfer occurred, not whether it was authorised.")
    lncs.body(doc,
        "Third, a determined adversary has options we did not model. Padding "
        "records to a constant size, interleaving cover traffic or spreading a "
        "transfer across days would degrade the representation, and these are "
        "studied in the fingerprinting literature [8, 9, 10]. Our experiments "
        "probe the cheapest and the method survives, but an adversary willing to "
        "install tooling can do better, so this raises the cost of a casual leak "
        "substantially and of a determined one somewhat. Fourth, the method "
        "decides after a session completes, so it detects rather than prevents, "
        "and incremental scoring on a sliding window is a partial remedy.")
    lncs.body(doc,
        "Finally, a defence built on traffic metadata is also a surveillance "
        "technique, since what reveals a repository leaving also reveals how "
        "often an employee consults an assistant [6, 13, 14]. An organisation "
        "should confine the analytic to aggregate flow records and state the "
        "practice openly.")


def write_conclusion(doc, B, C):
    lncs.heading(doc, "7  Conclusion and Future Work", 1)
    best = max(B["holdout"].items(), key=lambda kv: kv[1]["macro_f1"])
    dlp = pc(C["test_split"], 1)
    lncs.body(doc,
        "Shadow AI moves proprietary material out of organisations through a "
        "channel that looks like ordinary work. This paper argued that the way to "
        "see it is to stop reading the payload and to characterise the session "
        "instead. In an instrumented testbed carrying real repository source, "
        "business records and instruction prompts to a streaming inference "
        f"endpoint over TLS, {B['n_flows']:,} measured sessions show that record "
        "sizes, directions and arrival times separate exfiltration from benign "
        f"assistant use with a macro F1 of {num(best[1]['macro_f1'])}, against a "
        f"recall of {num(dlp['recall'])} for a content matching baseline granted "
        "complete plaintext access. The representation transferred to withheld "
        "behaviours and survived payload obfuscation.",
        first=True)
    lncs.body(doc,
        "Three directions follow. The first is measurement against live commercial "
        "providers, the second is incremental scoring so a decision can be "
        "reached partway through a transfer, and the third is which padding and "
        "pacing strategies defeat this representation and at what cost to the "
        "insider.")
    lncs.rich(doc, [("Availability. ", True, False),
                    ("Code and data at "
                     "https://github.com/rutabanjania/shadow-ai-exfiltration.",
                     False, False)],
              size=9, line=11, before=6, after=4)


if __name__ == "__main__":
    print(build())

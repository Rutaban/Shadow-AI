"""A content matching data loss prevention baseline.

The rule set follows what commodity DLP engines actually ship: identifiers with
fixed shapes, credential material and classification banners. It is applied to
the plaintext that the session uploads, which is the most favourable possible
setting for such an engine because it assumes complete decryption.
"""
from __future__ import annotations

import re

RULES = {
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
    "us_ssn": re.compile(r"\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\b(?:sk|pk)-[A-Za-z0-9]{20,}\b"),
    "iban": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,26}\b"),
    "classification_banner": re.compile(
        r"\b(?:CONFIDENTIAL|PROPRIETARY|INTERNAL USE ONLY|RESTRICTED)\b", re.IGNORECASE),
    "email_address": re.compile(r"\b[A-Za-z0-9._%%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_phone": re.compile(r"\b(?:\+1[ -]?)?\(?[2-9][0-9]{2}\)?[ -][0-9]{3}-[0-9]{4}\b"),
}

# Rules that a production deployment would treat as an alert rather than as
# telemetry. Email addresses alone are far too common to page an analyst on.
ALERTING = [
    "credit_card", "us_ssn", "aws_access_key", "private_key_block",
    "bearer_token", "iban", "classification_banner",
]


def scan(text: str) -> dict:
    hits = {name: len(rule.findall(text)) for name, rule in RULES.items()}
    hits["dlp_alert"] = int(any(hits[name] > 0 for name in ALERTING))
    return hits

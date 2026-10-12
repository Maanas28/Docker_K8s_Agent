"""Mask secret-looking values before file contents reach the LLM or the UI.

This is a safety net for secrets that ended up in ordinary files. Secret *files* (.env, keys)
are never read at all; see workspace.py.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

MASK = "[REDACTED]"

_SECRET_WORD = r"(?:secret|token|passw(?:or)?d|pwd|api[_-]?key|access[_-]?key|private[_-]?key|client[_-]?secret|auth)"

_RULES: list[tuple[re.Pattern[str], str]] = [
    # PEM private keys (whole block).
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        f"-----BEGIN PRIVATE KEY-----{MASK}-----END PRIVATE KEY-----",
    ),
    # Credentials inside URLs: scheme://user:password@host
    (re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]+)(@)", re.I), rf"\1{MASK}\3"),
    # Well-known token formats.
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), MASK),  # AWS access key id
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), MASK),  # GitHub
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), MASK),  # Slack
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), MASK),  # OpenAI/Anthropic-style keys
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), MASK),  # JWT
    # In code: a secret-named variable/field assigned a quoted string literal.
    #   API_KEY = "abc123"   password: 'hunter2'   "client_secret": "xyz"
    (
        re.compile(rf"""(?i)(["']?[\w.-]*{_SECRET_WORD}[\w.-]*["']?\s*[:=]\s*)(["'])([^"'\n]{{4,}})\2"""),
        rf"\1\2{MASK}\2",
    ),
]

# In config-style files, values are usually unquoted: `DB_PASSWORD=abc` or `password: abc`.
_CONFIG_LINE = re.compile(rf"(?im)^(\s*[\w.-]*{_SECRET_WORD}[\w.-]*\s*[:=]\s*)(?!\s*$)(?![\"'])(\S.*)$")
_CONFIG_SUFFIXES = {".ini", ".cfg", ".conf", ".properties", ".yaml", ".yml", ".toml", ".env"}


def _is_config(path: str) -> bool:
    p = PurePosixPath(path)
    return p.suffix.lower() in _CONFIG_SUFFIXES or p.name.lower().startswith(".env")


def redact(text: str, path: str = "") -> tuple[str, int]:
    """Return (masked text, number of values masked)."""
    count = 0
    for pattern, replacement in _RULES:
        text, n = pattern.subn(replacement, text)
        count += n
    if _is_config(path):
        text, n = _CONFIG_LINE.subn(rf"\1{MASK}", text)
        count += n
    return text, count

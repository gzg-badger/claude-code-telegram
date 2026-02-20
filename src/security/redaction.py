"""Output credential redaction.

Scans Claude responses for secret patterns and replaces them with [REDACTED]
before sending to Telegram. Prevents accidental credential leakage.
"""

import re
from typing import List, Pattern

# Compiled patterns for secret detection
SECRET_PATTERNS: List[Pattern] = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),           # Anthropic API key
    re.compile(r"xoxb-[A-Za-z0-9-]+"),                    # Slack bot token
    re.compile(r"xoxp-[A-Za-z0-9-]+"),                    # Slack user token
    re.compile(r"xoxe\.[A-Za-z0-9-]+"),                   # Slack refresh token
    re.compile(r"AKIA[0-9A-Z]{16}"),                       # AWS access key ID
    re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY"), # Private keys
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),                   # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36,}"),                   # GitHub OAuth
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),               # GitLab PAT
    re.compile(r"[0-9]+:[A-Za-z0-9_-]{35}"),               # Telegram bot token
    re.compile(r"re_[A-Za-z0-9]{20,}"),                    # Resend API key
    re.compile(r"AGE-SECRET-KEY-[A-Z0-9]+"),               # age encryption key
    re.compile(r"sk-[A-Za-z0-9]{32,}"),                    # Generic sk- API keys
    # JSON secrets - match "key": "value" patterns for sensitive keys
    re.compile(
        r'"(?:token|secret|key|password|api_key|apiKey|access_token|refresh_token|bot_token)"'
        r'\s*:\s*"[^"]{8,}"'
    ),
]


def redact_secrets(text: str) -> str:
    """Redact secrets from text before sending to Telegram.

    Args:
        text: The text to scan for secrets.

    Returns:
        Text with secrets replaced by [REDACTED].
    """
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text

"""Query classifier for three-tier model routing.

Routes messages to appropriate tiers:
- Tier 0 (instant): Greetings, acks — no Claude call needed
- Tier 1 (fast): Simple questions, lookups — Haiku + low effort
- Tier 2 (full): Everything else — Sonnet, current behavior
"""

import re
from typing import Optional, Tuple

# Tier 0: No Claude needed — instant bot response
INSTANT_PATTERNS = [
    r"^(hi|hello|hey|yo)[\s!.]*$",
    r"^(ok|okay|k|yep|yup|yes|no|nope)[\s!.]*$",
    r"^(thanks|thank you|thx|ty|cheers|ta)[\s!.]*$",
    r"^(got it|understood|noted|perfect|great|cool|nice|👍)[\s!.]*$",
    r"^(good morning|good evening|morning|evening|gm)[\s!.]*$",
]

# Tier 1: Fast — simple questions, lookups, status checks
FAST_PATTERNS = [
    r"^what('s| is) (the )?(time|date|day)",
    r"^(status|how('s| is) (the |)(vps|bot|system))",
    r"^who is\b",
    r"^what('s| is) .+ (email|role|title|team)",
    r"^(find|look up|search for)\b",
    r"^(where|when) (is|does|did)\b",
    r"^(how many|how much|count)\b",
    r"^(list|show) (my |the )?(tasks|meetings|events)",
    r"^what('s| is) on (my |)(calendar|schedule)",
    r"^(any|new) (messages|emails|notifications|alerts)",
    r"^(remind me|set a reminder)\b",
]

INSTANT_RESPONSES = {
    "greeting": "Hey George! What can I help with? \U0001f916\u26a1\ufe0f",
    "ack": "\U0001f44d",
    "thanks": "Anytime! \U0001f916",
}


def classify(message: str) -> Tuple[str, Optional[str]]:
    """Classify a message into a routing tier.

    Returns: ("instant"|"fast"|"full", optional_response)
    """
    text = message.strip().lower()

    # Check instant patterns
    for pattern in INSTANT_PATTERNS:
        if re.search(pattern, text):
            if re.search(r"(hi|hello|hey|yo|morning|evening|gm)", text):
                return ("instant", INSTANT_RESPONSES["greeting"])
            elif re.search(r"(thanks|thank|thx|ty|cheers|ta)", text):
                return ("instant", INSTANT_RESPONSES["thanks"])
            else:
                return ("instant", INSTANT_RESPONSES["ack"])

    # Check fast patterns
    for pattern in FAST_PATTERNS:
        if re.search(pattern, text):
            return ("fast", None)

    return ("full", None)

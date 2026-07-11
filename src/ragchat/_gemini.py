"""Shared Gemini generation call with 429 retry/backoff.

Free-tier quotas are small (a few requests/minute). Rather than crash on the
first 429, we wait out the limit — honouring the server's suggested retry delay
when it provides one. This is the kind of resilience real production callers need.
"""

from __future__ import annotations

import re
import time

from google.genai.errors import ClientError

_RETRY_DELAY_RE = re.compile(
    r"retry(?:Delay|\sin)['\":\s]+(\d+(?:\.\d+)?)s", re.IGNORECASE
)


def _suggested_delay(err: ClientError) -> float:
    m = _RETRY_DELAY_RE.search(str(err))
    return float(m.group(1)) if m else 0.0


def generate(client, *, max_retries: int = 6, **kwargs):
    """client.models.generate_content(**kwargs) with backoff on HTTP 429."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(**kwargs)
        except ClientError as err:
            if getattr(err, "code", None) != 429 or attempt == max_retries - 1:
                raise
            # honour the server's suggested delay (+buffer); else exponential backoff
            suggested = _suggested_delay(err)
            delay = suggested + 2 if suggested else min(20 * (attempt + 1), 60)
            time.sleep(delay)
    raise RuntimeError("unreachable")

"""Shared Gemini generation call with 429 retry/backoff.

Free-tier quotas are small (a few requests/minute). Rather than crash on the
first 429, we wait out the limit — honouring the server's suggested retry delay
when it provides one. This is the kind of resilience real production callers need.

Silent backoff is fine for a CLI; an interactive caller needs to know it's not
just hung. Pass `on_retry(delay, attempt)` to surface each wait before it sleeps.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator

from google.genai.errors import ClientError

_RETRY_DELAY_RE = re.compile(
    r"retry(?:Delay|\sin)['\":\s]+(\d+(?:\.\d+)?)s", re.IGNORECASE
)

OnRetry = Callable[[float, int], None]


def _suggested_delay(err: ClientError) -> float:
    m = _RETRY_DELAY_RE.search(str(err))
    return float(m.group(1)) if m else 0.0


def _backoff_delay(err: ClientError, attempt: int) -> float:
    suggested = _suggested_delay(err)
    return suggested + 2 if suggested else min(20 * (attempt + 1), 60)


def generate(
    client, *, max_retries: int = 6, on_retry: OnRetry | None = None, **kwargs
):
    """client.models.generate_content(**kwargs) with backoff on HTTP 429."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(**kwargs)
        except ClientError as err:
            if getattr(err, "code", None) != 429 or attempt == max_retries - 1:
                raise
            delay = _backoff_delay(err, attempt)
            if on_retry:
                on_retry(delay, attempt + 1)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def generate_stream(
    client, *, max_retries: int = 6, on_retry: OnRetry | None = None, **kwargs
) -> Iterator:
    """client.models.generate_content_stream(**kwargs) with backoff on HTTP 429.

    Retries only cover getting the FIRST chunk — once streaming has started,
    a mid-stream failure propagates rather than silently restarting (which
    would duplicate text already shown to the caller).
    """
    for attempt in range(max_retries):
        try:
            stream = client.models.generate_content_stream(**kwargs)
            first = next(stream)
        except StopIteration:
            return
        except ClientError as err:
            if getattr(err, "code", None) != 429 or attempt == max_retries - 1:
                raise
            delay = _backoff_delay(err, attempt)
            if on_retry:
                on_retry(delay, attempt + 1)
            time.sleep(delay)
            continue
        yield first
        yield from stream
        return
    raise RuntimeError("unreachable")

"""429 backoff: retry visibility and streaming, without real network calls."""

from google.genai.errors import ClientError

from ragchat import _gemini


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _RateLimitedThenOK:
    """A models.generate_content*-shaped fake: 429s N times, then succeeds."""

    def __init__(self, fail_times: int, chunks=None):
        self.fail_times = fail_times
        self.calls = 0
        self._chunks = chunks or [_FakeChunk("hello "), _FakeChunk("world")]

    def _maybe_fail(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ClientError(
                429, {"error": {"message": "RESOURCE_EXHAUSTED retryDelay: 0.01s"}}
            )

    def generate_content(self, **kwargs):
        self._maybe_fail()
        return _FakeChunk("full answer")

    def generate_content_stream(self, **kwargs):
        self._maybe_fail()
        return iter(self._chunks)


class _FakeClient:
    def __init__(self, fail_times: int, chunks=None):
        self.models = _RateLimitedThenOK(fail_times, chunks)


def test_generate_retries_and_calls_on_retry(monkeypatch):
    monkeypatch.setattr(_gemini.time, "sleep", lambda _: None)
    client = _FakeClient(fail_times=2)
    seen = []
    result = _gemini.generate(client, on_retry=lambda d, a: seen.append((d, a)))
    assert result.text == "full answer"
    assert [a for _, a in seen] == [1, 2]  # attempt numbers, in order


def test_generate_stream_retries_before_first_chunk(monkeypatch):
    monkeypatch.setattr(_gemini.time, "sleep", lambda _: None)
    client = _FakeClient(fail_times=1)
    seen = []
    chunks = list(
        _gemini.generate_stream(client, on_retry=lambda d, a: seen.append((d, a)))
    )
    assert [c.text for c in chunks] == ["hello ", "world"]
    assert seen == [(2.01, 1)]  # suggested 0.01s + 2s buffer, first attempt


def test_generate_stream_no_retry_needed_makes_no_sleep_call(monkeypatch):
    calls = []
    monkeypatch.setattr(_gemini.time, "sleep", lambda d: calls.append(d))
    client = _FakeClient(fail_times=0)
    chunks = list(_gemini.generate_stream(client))
    assert [c.text for c in chunks] == ["hello ", "world"]
    assert calls == []


def test_generate_reraises_non_429_immediately(monkeypatch):
    slept = []
    monkeypatch.setattr(_gemini.time, "sleep", lambda d: slept.append(d))

    class _Always403:
        def generate_content(self, **kwargs):
            raise ClientError(403, {"error": {"message": "PERMISSION_DENIED"}})

    class _C:
        models = _Always403()

    try:
        _gemini.generate(_C())
        raise AssertionError("expected ClientError")
    except ClientError as err:
        assert err.code == 403
    assert slept == []  # non-429 must not trigger backoff

"""Embedding functions accept an injected client — no network, no global state."""

from ragchat.embed import embed_documents, embed_query


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return type(
            "R", (), {"embeddings": [_FakeEmbedding([0.1, 0.2]) for _ in contents]}
        )()


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_embed_documents_uses_injected_client():
    client = _FakeClient()
    vectors = embed_documents(["chunk one", "chunk two"], client)
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    (call,) = client.models.calls
    assert call["contents"] == ["chunk one", "chunk two"]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_embed_query_uses_injected_client():
    client = _FakeClient()
    vector = embed_query("what is the refund window?", client)
    assert vector == [0.1, 0.2]
    (call,) = client.models.calls
    assert call["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_documents_empty_input_makes_no_api_call():
    client = _FakeClient()
    assert embed_documents([], client) == []
    assert client.models.calls == []

"""Shared fakes for the API tests: no network, no real keys, no real embedding model."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from ragbench.llm.client import LLMError
from ragbench.llm.fake import FakeLLMClient
from ragbench.metrics.cost import ModelPrice
from ragbench.questions import GoldFact, Question
from ragbench.types import Chunk, RetrievedChunk

# A syntactically plausible key, unique enough to grep for in logs and responses.
KEY = "gsk_" + "TESTKEYabc123" * 4
PRICING = {"fake-llm": ModelPrice(1.0, 2.0), "fake-judge": ModelPrice(0.1, 0.2)}
TITLES = {"d": "Document D"}
MODELS = {"generator": "fake-llm", "judge": "fake-judge"}


class HashEmbedder:
    name = "hash"

    def encode(self, texts: Any) -> Any:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def gold(text: str, quote: str) -> GoldFact:
    return GoldFact(text=text, doc_id="d", page=1, quote=quote)


QUESTIONS = [
    Question(id="L1", type="lookup", question="Lookup one?", gold_facts=[gold("f1", "alpha beta")]),
    Question(
        id="M1",
        type="multi_hop",
        question="Multi hop?",
        gold_facts=[gold("f3", "zeta eta"), gold("f4", "iota kappa")],
    ),
    Question(id="O1", type="out_of_corpus", question="Unknowable?", expected_abstain=True),
]


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, text: str, top_k: int = 4) -> list[RetrievedChunk]:
        self.queries.append(text)
        texts = ["alpha beta gamma", "unrelated text", "zeta eta theta", "more words"]
        return [
            RetrievedChunk(i + 1, 0.8 - 0.1 * i, Chunk(f"d:p1:c{i}", "d", "T", "core", 1, t))
            for i, t in enumerate(texts)
        ]


def judge_responder(messages: Any) -> str:
    text = messages[-1]["content"]
    if "Decompose the answer" in text:
        return json.dumps({"claims": ["c1", "c2"]})
    if "checking claims" in text:
        labels = ["supported", "unsupported"]
        return json.dumps(
            {
                "verdicts": [
                    {"claim_id": i + 1, "label": labels[i], "evidence_passage": 1, "reason": "why"}
                    for i in range(2)
                ]
            }
        )
    n = len(re.findall(r"^\d+\. ", text.split("Gold facts:")[-1], re.M))
    return json.dumps({"coverage": [{"fact_id": i + 1, "covered": i == 0} for i in range(n)]})


class FakeFactory:
    """A ClientFactory that records every (role, api_key) it is asked for."""

    def __init__(
        self, *, generator: FakeLLMClient | None = None, judge: FakeLLMClient | None = None
    ):
        self.generator = generator or FakeLLMClient()
        self.judge = judge or FakeLLMClient(judge_responder, model="fake-judge")
        self.requests: list[tuple[str, str]] = []

    def __call__(self, role: str, api_key: str) -> FakeLLMClient:
        self.requests.append((role, api_key))
        return self.generator if role == "generator" else self.judge


class FailingClient(FakeLLMClient):
    """Raises a chosen exception on every chat call."""

    def __init__(self, exc: Exception, model: str = "fake-llm") -> None:
        super().__init__(model=model)
        self._exc = exc

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        raise self._exc


def llm_error(code: str, message: str = "provider failure") -> LLMError:
    return LLMError(message, code=code)


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a server-sent-events body into (event, payload) pairs."""
    events = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        name, data = None, []
        for line in re.split(r"\r?\n", block):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events

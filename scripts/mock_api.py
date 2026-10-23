"""A dev mock of the live API: the real FastAPI app with a fake model, no network, no key checks.

    python scripts/mock_api.py [port]     # default http://127.0.0.1:8000

Any plausible key (gsk_ plus 20+ characters) is accepted. The retriever is real. The fake models
are crude but *discriminating*, so the UI's scores vary the way real ones do instead of showing one
constant for every condition:

- the generator quotes the first retrieved passage (and answers the no-retrieval control with a
  generic sentence that is not in the corpus);
- the judge checks each claim by word overlap with the passages, and each gold fact by word
  overlap with the answer.

The models are named "mock-generator" / "mock-judge", and the UI shows those names on live results.
These scores mean nothing about answer quality; use a real key for that.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

import uvicorn

from ragbench.api.main import create_app
from ragbench.api.settings import ApiSettings
from ragbench.llm.fake import FakeLLMClient, default_responder

DELAY_S = 0.9  # per model call, so progress states are visible in the UI
_STOP = {"that", "this", "with", "from", "they", "their", "which", "were", "have", "been", "into"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def _overlap(needle: str, haystack: str) -> float:
    words = _content_words(needle)
    return len(words & _content_words(haystack)) / len(words) if words else 0.0


def _passage_quote(prompt: str) -> str:
    """One sentence from each of the first three passages, so the answer spans the context."""
    parts = re.findall(r"\[(\d)\]\s*(.+?)(?=\n\s*\[\d\]|\Z)", prompt, re.S)
    sentences = [re.split(r"(?<=[.!?])\s+", body.strip())[0] for _, body in parts[:3]]
    return " ".join(sentences)[:600] or "The passages do not say."


def generator_responder(messages: Any) -> str:
    text = "\n".join(m["content"] for m in messages)
    base = default_responder(messages)
    if "strict reviewer" in text or "reading retrieved passages" in text:
        return base  # critique / extraction stages keep the shape the strategy parses
    if "No reference documents are provided" in text:
        return "Retrieval-augmented systems combine a retriever with a language model."
    quote = _passage_quote(text)
    if "## Summary" in text:
        points = f"- {quote}\n- two\n- three"
        return f"## Summary\n{quote}\n\n## Key Points\n{points}\n\n## Elaboration\n{quote}"
    return quote


def judge_responder(messages: Any) -> str:
    text = messages[-1]["content"]
    if "Decompose the answer" in text:
        answer = text.split("Answer:", 1)[1].split("Return JSON only", 1)[0]
        claims = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s.strip()) > 12
        ]
        return json.dumps({"claims": claims[:4]})
    if "checking claims" in text:
        context = text.split("Passages:", 1)[1].split("Claims:", 1)[0]
        claims = re.findall(
            r"^\d+\.\s*(.+)$", text.split("Claims:", 1)[1].split("For each claim", 1)[0], re.M
        )
        verdicts = []
        for i, claim in enumerate(claims, start=1):
            ok = _overlap(claim, context) >= 0.6
            verdicts.append(
                {
                    "claim_id": i,
                    "label": "supported" if ok else "unsupported",
                    "evidence_passage": 1 if ok else None,
                    "reason": "Words found in the passages."
                    if ok
                    else "Not found in the passages.",
                }
            )
        return json.dumps({"verdicts": verdicts})
    answer = text.split("Answer:", 1)[1].split("Gold facts:", 1)[0]
    facts = re.findall(r"^\d+\.\s*(.+)$", text.split("Gold facts:", 1)[1], re.M)
    return json.dumps(
        {
            "coverage": [
                {"fact_id": i, "covered": _overlap(f, answer) >= 0.4}
                for i, f in enumerate(facts, 1)
            ]
        }
    )


class SlowFake(FakeLLMClient):
    def chat(self, messages: Any, **kwargs: Any) -> Any:
        time.sleep(DELAY_S)
        return super().chat(messages, **kwargs)


def factory(role: str, api_key: str) -> FakeLLMClient:
    if role == "generator":
        return SlowFake(generator_responder, model="mock-generator")
    return SlowFake(judge_responder, model="mock-judge")


app = create_app(
    settings=ApiSettings(_env_file=None, rate_limit_runs=1000, rate_limit_validate=1000),
    client_factory=factory,
    models={"generator": "mock-generator", "judge": "mock-judge"},
)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="127.0.0.1", port=port)

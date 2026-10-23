"""The benchmark question set (data/questions.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from ragbench.config import repo_root

QuestionType = Literal["lookup", "multi_hop", "out_of_corpus"]


class GoldFact(BaseModel):
    text: str
    doc_id: str
    page: int
    quote: str  # verbatim from the cleaned corpus page; a test enforces it


class Question(BaseModel):
    id: str
    type: QuestionType
    question: str
    gold_facts: list[GoldFact] = []
    expected_abstain: bool = False
    notes: str | None = None

    @property
    def gold_fact_texts(self) -> list[str]:
        return [g.text for g in self.gold_facts]


class DroppedQuestion(BaseModel):
    id: str
    type: QuestionType
    question: str
    reason: str


class QuestionSet(BaseModel):
    description: str = ""
    questions: list[Question]
    dropped_questions: list[DroppedQuestion] = []

    def by_id(self, qid: str) -> Question:
        return next(q for q in self.questions if q.id == qid)


def load_questions(path: Path | None = None) -> QuestionSet:
    path = path or repo_root() / "data" / "questions.yaml"
    with path.open(encoding="utf-8") as fh:
        return QuestionSet.model_validate(yaml.safe_load(fh))

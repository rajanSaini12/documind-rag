"""Loading the evaluation Q&A dataset (JSON Lines, one example per line)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class QAExample(BaseModel):
    id: str
    question: str
    answerable: bool = True
    reference_answer: str | None = Field(
        default=None, description="Gold answer written by a human; required if answerable."
    )
    source: str | None = Field(default=None, description="File name containing the answer.")
    gold_pages: list[int] = Field(default_factory=list, description="1-based page numbers.")

    @model_validator(mode="after")
    def _check_answerable_fields(self) -> QAExample:
        if self.answerable and not (self.reference_answer and self.source and self.gold_pages):
            raise ValueError(
                f"{self.id}: answerable examples need reference_answer, source and gold_pages"
            )
        return self


def load_dataset(path: str | Path) -> list[QAExample]:
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(QAExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid example: {exc}") from exc
    return examples

"""Read the hard constraints out of a scouting question.

Embedding search ranks by overall resemblance, which is the wrong tool for an absolute
requirement: "left-footed" is one token in a profile of dozens and never dominates the
similarity, so a semantic search for "a young left-footed winger" happily returns
right-footed ones. These constraints are pulled out first and used to narrow the pool
*before* ranking it.

Deliberately deterministic rather than a second LLM call: it costs nothing, it cannot
hallucinate a filter the user did not ask for, and it is testable. The cost is that only
the phrasings listed here are understood — anything else yields no constraint and
retrieval behaves as it did before, which is the safe direction to fail in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "young" has no agreed definition; this is the usual scouting shorthand, and the answer
# states which threshold was applied so the reader is not left guessing.
YOUNG_MAX_AGE = 23.0
VETERAN_MIN_AGE = 32.0


@dataclass(frozen=True)
class Constraints:
    foot: str | None = None
    position_group: str | None = None
    max_age: float | None = None
    min_age: float | None = None

    def __bool__(self) -> bool:
        return any(
            v is not None for v in (self.foot, self.position_group, self.max_age, self.min_age)
        )

    def describe(self) -> str:
        """Human-readable summary, so the answer can state what it filtered on."""
        parts = []
        if self.foot:
            parts.append("two-footed" if self.foot == "both" else f"{self.foot}-footed")
        if self.position_group:
            parts.append(self.position_group)
        if self.max_age is not None:
            parts.append(f"aged {self.max_age:.0f} or under")
        if self.min_age is not None:
            parts.append(f"aged {self.min_age:.0f} or over")
        return ", ".join(parts)


_FOOT_PATTERNS = (
    (re.compile(r"\b(left[\s-]?footed|left foot)\b"), "left"),
    (re.compile(r"\b(right[\s-]?footed|right foot)\b"), "right"),
    (re.compile(r"\b(two[\s-]?footed|both feet|ambidextrous)\b"), "both"),
)

# Matched by earliest mention, not by list order. A question names its subject before its
# object — "a winger who takes defenders on" is about the winger — so the first position
# word in the sentence is the one being asked for. Within each pattern the compound forms
# come first so "centre-back" is not read as a forward on the word "centre".
_POSITION_PATTERNS = (
    (re.compile(r"\b(goalkeeper|keeper|goalie)s?\b"), "GK"),
    (
        re.compile(
            r"\b(wing[\s-]?back|full[\s-]?back|centre[\s-]?back|center[\s-]?back|"
            r"defender|defence|defense)s?\b"
        ),
        "DEF",
    ),
    (re.compile(r"\b(midfield(er)?|playmaker|holding|regista)s?\b"), "MID"),
    (re.compile(r"\b(winger|striker|forward|attacker|centre[\s-]?forward)s?\b"), "FWD"),
)

_UNDER = re.compile(r"\b(?:under|younger than|below|u)[\s-]?(\d{2})\b")
_OVER = re.compile(r"\b(?:over|older than|above)[\s-]?(\d{2})\b")
_YOUNG = re.compile(r"\b(young|youngster|prospect|teenager)s?\b")
_VETERAN = re.compile(r"\b(veteran|experienced|older)s?\b")


def parse_constraints(question: str) -> Constraints:
    """Pull the absolute requirements out of a question; everything else is left to rank."""
    text = question.lower()

    foot = next((value for pattern, value in _FOOT_PATTERNS if pattern.search(text)), None)

    found = [
        (match.start(), value)
        for pattern, value in _POSITION_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    position = min(found)[1] if found else None

    max_age: float | None = None
    min_age: float | None = None
    if (explicit := _UNDER.search(text)) is not None:
        max_age = float(explicit.group(1))
    elif _YOUNG.search(text):
        max_age = YOUNG_MAX_AGE
    if (explicit := _OVER.search(text)) is not None:
        min_age = float(explicit.group(1))
    elif _VETERAN.search(text):
        min_age = VETERAN_MIN_AGE

    return Constraints(foot=foot, position_group=position, max_age=max_age, min_age=min_age)

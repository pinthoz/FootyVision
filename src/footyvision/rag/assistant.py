"""RAG scouting assistant: retrieve relevant player profiles, then answer grounded in them."""

from __future__ import annotations

from typing import Any

from footyvision.llm.client import LLMClient
from footyvision.rag.constraints import Constraints, parse_constraints
from footyvision.rag.metrics import metrics_in
from footyvision.rag.store import VectorStore


def build_prompt(
    question: str,
    hits: list,
    constraints: Constraints | None = None,
    unfilterable: dict[str, int] | None = None,
    metric_notes: list[str] | None = None,
) -> tuple[str, str]:
    system = (
        "You are a football scouting assistant. Answer in English, concisely, even when "
        "the question is asked in another language. Base your answer EXCLUSIVELY on the "
        "retrieved players below — never invent players or statistics. Name the players "
        "you use, and if none of them fit, say so honestly. Every profile carries per-90 "
        "values and percentiles: when the question compares players or asks who is best "
        "at some metric, compare those numbers and justify your pick with them. If the "
        "metric asked about is not in the profiles, say you do not have it rather than "
        "guessing."
    )
    if constraints:
        # The pool was narrowed before ranking, so the model must not re-litigate the
        # filter — every candidate shown already satisfies it, and none of them should be
        # rejected for failing a requirement they were selected on.
        system += (
            " The retrieved players have ALREADY been filtered to those who are "
            f"{constraints.describe()}, so treat that requirement as met by all of them "
            "and judge only the rest of the question. If the list is empty, say plainly "
            "that no player in the dataset meets the requirement."
        )
    if unfilterable:
        # Stated as a limit on the search, not as a fact about the players: the shortlist
        # is drawn from a smaller pool than the reader assumes, and that changes what
        # "the best available" means.
        missing = ", ".join(
            f"{count} players have no recorded {field}" for field, count in unfilterable.items()
        )
        system += (
            f" Note that {missing}, so they could not be checked against this requirement "
            "and were left out of the search. End your answer with one short sentence "
            "saying so, phrased as a limit of the data rather than a judgement on them."
            # Without this the model reads its own shortlist as a census. Asked for young
            # players in Liga F it answered "no players in the dataset play in Liga F" —
            # false, there are 275 of them; the age filter had removed every one, because
            # no woman in this data has a recorded date of birth. The retrieved six were
            # men, and it described them accurately and concluded something untrue.
            " The players shown are what survived that filter, NOT the whole database. "
            "Never say a competition, country or group is absent from the dataset merely "
            "because none of the retrieved players belongs to it — say instead that the "
            "requirement could not be checked for them."
        )
    context = "\n".join(f"- {h.text}" for h in hits) or "(no player matches the requirement)"
    user = f"Question: {question}\n\nRetrieved players (context):\n{context}\n"
    if metric_notes:
        # Listed apart from the profiles and labelled, so the model reads them as the
        # answer to what was asked rather than as another paragraph of style prose.
        listed = "\n".join(f"- {note}" for note in metric_notes)
        user += f"\nThe metric asked about, for those same players:\n{listed}\n"
    user += "\nAnswer:"
    return system, user


class ScoutAssistant:
    def __init__(self, store: VectorStore, client: LLMClient | None = None) -> None:
        self.store = store
        self.client = client or LLMClient()

    def answer(self, question: str, k: int = 6) -> dict[str, Any]:
        # Hybrid retrieval: players named in the question are pinned, the rest of the
        # budget is filled by embedding similarity.
        # Hard requirements ("left-footed", "under 23") narrow the pool before ranking:
        # an embedding cannot enforce them, it can only prefer them, and preference is
        # not enough when the constraint is absolute.
        # The index supplies the nationality vocabulary, so only countries someone in the
        # pool actually comes from can become a filter.
        constraints = parse_constraints(question, self.store.countries)
        mask = self.store.matching(constraints) if constraints else None
        # Players the filter removed because the attribute is missing for them, not
        # because they failed it. Reported rather than swallowed: age and foot are only
        # known for the men's leagues, so those two requirements quietly take the women's
        # competitions out of the running, and a silent omission reads as an answer.
        unfilterable = self.store.unknown_dropped(constraints) if constraints else {}

        pinned = self.store.mentioned(question)
        # When the question names players, fill the remaining slots with players whose
        # *style* resembles theirs. Embedding the raw question makes the proper nouns
        # dominate, which retrieves similar-sounding names instead of similar players.
        centroid = self.store.style_centroid([h.player_id for h in pinned]) if pinned else None
        query_vector = (
            centroid if centroid is not None else self.client.embed([question], kind="query")[0]
        )
        # Under a hard constraint the pinned players are the *seed*, not answers: a named
        # player need not satisfy the requirement ("left-footed wingers like Bale" — Bale
        # is right-footed), and showing him would contradict the promise the prompt makes
        # that every listed player already passes the filter.
        hits = [] if constraints else list(pinned)
        seen = {h.player_id for h in hits}
        for hit in self.store.search(query_vector, k=k + len(hits), mask=mask):
            if len(hits) >= max(k, len(hits)):
                break
            if hit.player_id not in seen:
                hits.append(hit)
                seen.add(hit.player_id)
        # A profile names a player's four best metrics and their weakest. Asked about any
        # other one the model used to answer that it had no such data — true of what it
        # had been given, false of what the database holds. The metric named in the
        # question is looked up and handed over with the profiles.
        asked = metrics_in(question)
        notes = self.store.metric_notes(hits, asked)
        contexts = [h.text for h in hits] + notes

        system, user = build_prompt(question, hits, constraints, unfilterable, notes)
        answer = self.client.chat(system, user, max_tokens=1400)
        return {
            "answer": answer,
            "filters": constraints.describe() or None,
            "not_considered": unfilterable or None,
            # The profiles the answer was written from. Absent from `AssistantResponse`,
            # so the HTTP payload is unchanged — this is here because an evaluation of
            # faithfulness needs the exact text the model was grounded in, and rebuilding
            # it afterwards from player ids would be measuring a different retrieval.
            "contexts": contexts,
            "sources": [
                {"player_id": h.player_id, "name": h.name, "score": round(h.score, 3)} for h in hits
            ],
        }

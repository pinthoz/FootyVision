"""Unit tests for the RAG assistant — synthetic data, no DB or live LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.rag.assistant import ScoutAssistant
from footyvision.rag.profiles import build_profiles
from footyvision.rag.store import VectorStore


def _frame() -> pd.DataFrame:
    rows = []
    specs = {"Center Forward": {"xg_per90": 0.6}, "Center Back": {"tackles_per90": 3.0}}
    pid = 1
    for position, base in specs.items():
        for _ in range(3):
            feats = {f: 0.0 for f in PER90_FEATURES}
            feats.update(base)
            rows.append(
                {
                    "player_id": pid,
                    "name": f"Player {pid}",
                    "competition": "La Liga",
                    "primary_position": position,
                    "position_group": position_group(position),
                    "minutes": 1800,
                    "matches_played": 20,
                    **feats,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_build_profiles_text():
    docs = build_profiles(_frame())
    assert len(docs) == 6
    d = docs[0]
    assert d["name"] in d["text"]
    assert "Performance score" in d["text"]


class _FakeClient:
    """embed maps text/query to a 2D vector via a keyword; chat echoes the context."""

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        self.last_kind = kind
        return [[1.0, 0.0] if "forward" in t.lower() else [0.0, 1.0] for t in texts]

    def chat(self, system: str, user: str, **_) -> str:
        return "Answer grounded in the retrieved players."


def test_vector_store_build_and_search():
    docs = [
        {"player_id": 1, "name": "Striker", "text": "a lethal forward"},
        {"player_id": 2, "name": "Defender", "text": "a solid centre back"},
    ]
    store = VectorStore.build(docs, _FakeClient())
    assert len(store) == 2
    # Query embeds to the "forward" vector -> Striker ranks first.
    hits = store.search([1.0, 0.0], k=2)
    assert hits[0].name == "Striker"
    assert hits[0].score >= hits[1].score


def test_assistant_answer_grounds_and_cites():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(
        [1, 2], ["Striker", "Defender"], ["a lethal forward", "a solid centre back"], vectors
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("who is a good forward?", k=1)
    assert "Answer" in result["answer"]
    assert result["sources"][0]["name"] == "Striker"


def test_profiles_carry_the_numbers_behind_each_phrase():
    # Without values the assistant cannot answer "who has the most xG?" — every profile
    # collapses to the same stock phrases.
    docs = build_profiles(_frame())
    forward = next(d for d in docs if "Center Forward" in d["text"])
    assert "0.60 xG per 90" in forward["text"]
    assert "percentile" in forward["text"]


def test_store_pins_players_named_in_the_question():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(
        [1, 2],
        ["Gareth Frank Bale", "Cédric Bakambu"],
        ["a winger", "a striker"],
        vectors,
    )
    # Accent-insensitive, and short words in the question must not match anything.
    named = {h.name for h in store.mentioned("most xG like Bale and bakambu?")}
    assert named == {"Gareth Frank Bale", "Cédric Bakambu"}
    assert store.mentioned("who is the best midfielder?") == []


def test_assistant_retrieves_named_players_even_when_embeddings_disagree():
    # Both stored vectors are orthogonal to the query embedding, so pure similarity
    # search would surface whichever happens to rank first — not necessarily the two
    # players the question is about.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(
        [1, 2, 3],
        ["Gareth Frank Bale", "Cédric Bakambu", "Someone Else"],
        ["a winger", "a striker", "a defender"],
        np.vstack([vectors, [[0.0, 1.0]]]).astype(np.float32),
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("xG like Bale and Bakambu?", k=1)
    names = {s["name"] for s in result["sources"]}
    assert {"Gareth Frank Bale", "Cédric Bakambu"} <= names


def test_named_players_pull_in_stylistic_neighbours_not_similar_names():
    # Two forwards share a style vector; the third is a defender whose *name* is close to
    # the ones in the question. Retrieval must follow style, not spelling.
    vectors = np.array(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]],
        dtype=np.float32,
    )
    store = VectorStore(
        [1, 2, 3],
        ["Cédric Bakambu", "Another Forward", "Alhassane Bangoura"],
        ["a striker", "another striker", "a defender"],
        vectors,
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("most xG like Bakambu?", k=2)
    names = [s["name"] for s in result["sources"]]
    assert names[0] == "Cédric Bakambu"
    assert "Another Forward" in names
    assert "Alhassane Bangoura" not in names


def test_client_applies_the_query_prefix_to_questions():
    # Retrieval models are trained asymmetrically: a question must carry the query prefix,
    # indexed profiles the document one. Embedding both the same way measurably hurts —
    # see scripts/eval_embeddings.py.
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    store = VectorStore([1], ["Striker"], ["a lethal forward"], vectors)
    client = _FakeClient()
    ScoutAssistant(store, client=client).answer("who scores goals?", k=1)
    assert client.last_kind == "query"


# --- lexical pinning vs ordinary words --------------------------------------------------


def _store_with(names):
    import numpy as np

    from footyvision.rag.store import VectorStore

    return VectorStore(
        list(range(1, len(names) + 1)),
        names,
        [f"{n} profile" for n in names],
        np.eye(len(names), 4, dtype=np.float32),
    )


def test_an_ordinary_word_does_not_pin_a_player_who_shares_it():
    """ "a young winger" must not pin Ashley Young.

    The assistant queries with the centroid of the pinned players rather than the
    question, so one accidental name collision would hijack the entire retrieval.
    """
    store = _store_with(["Ashley Young", "Gareth Frank Bale"])

    assert store.mentioned("find me a young left-footed winger") == []


def test_a_player_whose_surname_is_an_ordinary_word_is_still_findable():
    """Filtering the question side is safe: the rest of the name still identifies him."""
    store = _store_with(["Ashley Young", "Gareth Frank Bale"])

    assert [h.name for h in store.mentioned("how good is Ashley Young?")] == ["Ashley Young"]


def test_pinning_still_finds_the_players_a_comparison_names():
    store = _store_with(["Gareth Frank Bale", "Cedric Bakambu", "Ashley Young"])

    pinned = {h.name for h in store.mentioned("compare Bale and Bakambu")}

    assert pinned == {"Gareth Frank Bale", "Cedric Bakambu"}


# --- hard constraints before ranking ----------------------------------------------------


def test_constraints_read_the_absolute_requirements_from_a_question():
    from footyvision.rag.constraints import parse_constraints

    c = parse_constraints("find me a young left-footed winger who takes defenders on")

    assert c.foot == "left"
    # "winger" is the subject, "defenders" the object: earliest mention wins.
    assert c.position_group == "FWD"
    assert c.max_age == 23.0


def test_constraints_prefer_an_explicit_age_over_the_word_young():
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints("young midfielder under 21").max_age == 21.0


def test_constraints_are_empty_for_a_question_that_states_none():
    from footyvision.rag.constraints import parse_constraints

    c = parse_constraints("who has the most interceptions?")

    assert not c
    assert c.describe() == ""


def _attr_store():
    import numpy as np

    from footyvision.rag.store import VectorStore

    names = ["Lefty Young", "Righty Young", "Lefty Old", "Unknown Foot"]
    return VectorStore(
        [1, 2, 3, 4],
        names,
        [f"{n} profile" for n in names],
        np.eye(4, 4, dtype=np.float32),
        attrs={
            "foot": ["left", "right", "left", None],
            "age": [21.0, 21.0, 33.0, 21.0],
            "position_group": ["FWD", "FWD", "FWD", "FWD"],
        },
    )


def test_matching_narrows_the_pool_to_the_constraint():
    from footyvision.rag.constraints import Constraints

    store = _attr_store()

    assert list(store.ids[store.matching(Constraints(foot="left", max_age=23.0))]) == [1]


def test_an_unknown_attribute_fails_the_constraint_rather_than_passing_it():
    """Calling a player left-footed when we never recorded his foot is an invention."""
    from footyvision.rag.constraints import Constraints

    store = _attr_store()

    assert list(store.ids[store.matching(Constraints(foot="left"))]) == [1, 3]


def test_search_ranks_only_within_the_filtered_subset():
    from footyvision.rag.constraints import Constraints

    store = _attr_store()
    # Closest to player 2, who fails the filter. Players 1 and 3 tie at 0, so only
    # membership is asserted — the tie order is arbitrary.
    hits = store.search([0, 1, 0, 0], k=3, mask=store.matching(Constraints(foot="left")))

    assert {h.player_id for h in hits} == {1, 3}


def test_search_returns_nothing_when_the_constraint_excludes_everyone():
    from footyvision.rag.constraints import Constraints

    store = _attr_store()

    assert store.search([1, 0, 0, 0], k=3, mask=store.matching(Constraints(foot="both"))) == []


def test_a_dimension_the_index_knows_nothing_about_is_not_a_filter():
    """Otherwise an index built before these attributes existed excludes everybody."""
    import numpy as np

    from footyvision.rag.constraints import Constraints
    from footyvision.rag.store import VectorStore

    store = VectorStore([1, 2], ["A", "B"], ["a", "b"], np.eye(2, 4, dtype=np.float32))

    assert store.matching(Constraints(foot="left", position_group="FWD")).all()


def test_prompt_tells_the_model_the_filter_is_already_applied():
    from footyvision.rag.assistant import build_prompt
    from footyvision.rag.constraints import Constraints

    system, _ = build_prompt("a left-footed winger", [], Constraints(foot="left"))

    assert "ALREADY been filtered" in system
    assert "left-footed" in system


def test_prompt_says_so_when_no_player_matches():
    from footyvision.rag.assistant import build_prompt
    from footyvision.rag.constraints import Constraints

    _, user = build_prompt("a two-footed keeper", [], Constraints(foot="both"))

    assert "no player matches the requirement" in user


# --- profile biography and index staleness ----------------------------------------------


def test_profile_mentions_age_foot_and_height_when_known(db_session):
    """Scouts ask for "a young left-footed winger"; the text has to contain those words."""
    from footyvision.ml.features import load_feature_frame
    from footyvision.rag.profiles import build_profiles

    frame = load_feature_frame(db_session, 600)
    frame["age"] = 24.0
    frame["foot"] = "left"
    frame["height_cm"] = 185.0

    text = build_profiles(frame)[0]["text"]

    assert "24 years old" in text
    assert "left-footed" in text
    assert "1.85m tall" in text


def test_profile_omits_details_it_does_not_have(db_session):
    """Roughly a tenth of players have no date of birth or foot; the text must not lie."""
    import numpy as np

    from footyvision.ml.features import load_feature_frame
    from footyvision.rag.profiles import build_profiles

    frame = load_feature_frame(db_session, 600)
    frame["age"] = np.nan
    frame["foot"] = None
    frame["height_cm"] = np.nan

    text = build_profiles(frame)[0]["text"]

    assert "years old" not in text
    assert "footed" not in text


def test_store_is_stale_reports_players_missing_from_the_index(db_session):
    """A stale index fails silently, so the gap has to be measurable."""
    import numpy as np

    from footyvision.rag.service import store_is_stale
    from footyvision.rag.store import VectorStore

    store = VectorStore([1], ["Alpha Striker"], ["a profile"], np.ones((1, 4), dtype=np.float32))

    assert store_is_stale(db_session, store) > 0

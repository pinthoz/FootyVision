"""Unit tests for the RAG assistant — synthetic data, no DB or live LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_a_name_many_players_share_pins_none_of_them():
    """ "Compare Neymar and Luis Suarez" used to pin nineteen players.

    Every one of them carried "Luis" or "Suarez" somewhere in their name, and the
    assistant then queried with the centroid of all nineteen and answered about none. A
    name part belonging to more than a handful of people identifies a family, not a
    person — and which parts those are is a property of who was loaded, so it is counted
    off the index rather than listed by hand.
    """
    from footyvision.rag.store import MAX_PLAYERS_PER_TOKEN

    shared = [f"Luis Silva {i}" for i in range(MAX_PLAYERS_PER_TOKEN + 1)]
    store = _store_with([*shared, "Gareth Frank Bale"])

    assert store.mentioned("compare Luis and Bale") == [h for h in store.mentioned("Bale")]
    # One below the threshold is still distinctive enough to pin on.
    fewer = _store_with([f"Luis Silva {i}" for i in range(MAX_PLAYERS_PER_TOKEN)])
    assert len(fewer.mentioned("how good is Luis?")) == MAX_PLAYERS_PER_TOKEN


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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # The pair that exposed the gap: one was filtered, the other was not.
        ("a left-footed winger", "left-footed, FWD"),
        ("um extremo canhoto", "left-footed, FWD"),
        # Portuguese inflects the adjective, which a bare `s?` on an English stem misses.
        ("avancados ambidestros", "two-footed, FWD"),
        # ...and pluralises "-al" as "-ais", which it cannot build at all.
        ("laterais destros", "right-footed, DEF"),
        ("defesas centrais", "DEF"),
        ("guarda-redes experientes", "GK, aged 32 or over"),
        ("ponta de lanca sub-21", "FWD, aged 21 or under"),
        ("medios com mais de 30 anos", "MID, aged 30 or over"),
        # "medio ala" is a midfielder: the first position word in the sentence wins, and
        # "ala" alone would otherwise be read as a wing-back.
        ("um medio ala jovem", "MID, aged 23 or under"),
        ("quem e o melhor jogador", ""),
    ],
)
def test_constraints_are_read_in_portuguese_as_well_as_english(question, expected):
    """The assistant takes questions in any language; the filters must too.

    While these patterns were English-only, "a left-footed winger" was filtered down to
    left-footed forwards and "um extremo canhoto" was not filtered at all — it fell back
    to similarity ranking, which cannot enforce a foot and will happily return right-footed
    players. A silent difference in behaviour between two ways of asking the same thing.
    """
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints(question).describe() == expected


def test_feminine_forms_are_read_too():
    """Half the pool is women's football, and Portuguese inflects for gender."""
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints("avancadas canhotas").describe() == "left-footed, FWD"
    assert parse_constraints("uma lateral destra").describe() == "right-footed, DEF"
    assert parse_constraints("jogadoras ambidestras").foot == "both"


def test_an_ambiguous_word_is_left_unparsed_rather_than_guessed():
    """ "médias" is both female midfielders and averages, and the accents are stripped.

    Reading it as a position would filter "qual e a media de remates" down to midfielders
    and answer a question nobody asked. Not filtering costs a narrowing; filtering wrongly
    costs the answer.
    """
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints("qual e a media de remates").position_group is None
    # The male form is unambiguous and still parses.
    assert parse_constraints("medios que pressionam alto").position_group == "MID"


_POOL = ["Spain", "Brazil", "Portugal", "Côte d'Ivoire", "Venezuela\xa0(Bolivarian Republic)"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("a Brazilian left-footed winger", "left-footed, FWD, from Brazil"),
        ("um extremo brasileiro canhoto", "left-footed, FWD, from Brazil"),
        # Feminine and plural, because half the pool is women's football.
        ("avancadas brasileiras", "FWD, from Brazil"),
        ("uma lateral portuguesa", "DEF, from Portugal"),
        # "espanhol" pluralises irregularly, to "espanhóis".
        ("medios espanhois", "MID, from Spain"),
        # Stored with an accent; asked without one.
        ("wingers from Ivory Coast", "FWD, from Côte d'Ivoire"),
        ("extremos da costa do marfim", "FWD, from Côte d'Ivoire"),
        # Stored with a parenthetical and a non-breaking space; asked as the plain name.
        ("players from Venezuela", "from Venezuela\xa0(Bolivarian Republic)"),
    ],
)
def test_nationality_is_read_and_resolved_to_the_stored_spelling(question, expected):
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints(question, _POOL).describe() == expected


def test_a_country_absent_from_the_pool_is_not_a_filter():
    """Narrowing to a country nobody comes from would answer "no such player exists".

    That is a different claim from "none was ever loaded", and the second is the true one.
    Falling back to ranking at least returns the closest thing the data holds.
    """
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints("a Nigerian striker", _POOL).nationality is None
    # Without the index's vocabulary there is no nationality filter at all.
    assert parse_constraints("a Brazilian striker").nationality is None


def test_the_store_offers_the_nationalities_it_holds():
    import numpy as np

    from footyvision.rag.store import VectorStore

    store = VectorStore(
        [1, 2, 3],
        ["A", "B", "C"],
        ["a", "b", "c"],
        np.eye(3, 4, dtype=np.float32),
        attrs={"nationality": ["Brazil", "Spain", None]},
    )

    assert store.countries == ["Brazil", "Spain"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # The question that exposed the gap: the assistant answered that it had no tackle
        # data, while `tackles_per90` sat in the database for every player.
        ("Quais os medios experientes com mais desarmes?", ["tackles_per90"]),
        ("who makes the most tackles and interceptions?", ["tackles_per90", "interceptions_per90"]),
        # A compound phrase must beat the simple one inside it, in both directions.
        ("quem tem mais passes progressivos", ["progressive_passes_per90"]),
        ("melhores em dribles completos", ["dribbles_completed_per90"]),
        ("progressive carries", ["progressive_carries_per90"]),
        # Portuguese names this one as a verb far more often than as a noun.
        ("quais os guarda-redes que mais recuperam bolas?", ["ball_recoveries_per90"]),
        # Nothing we hold: better no note than a guessed one.
        ("who wins the most aerial duels?", []),
        ("quem e o melhor jogador", []),
    ],
)
def test_the_metric_a_question_asks_about_is_read_from_the_question(question, expected):
    from footyvision.rag.metrics import metrics_in

    assert metrics_in(question) == expected


def test_metric_notes_supply_values_the_profile_prose_leaves_out():
    """A profile names four strengths and one weakness; the other twelve are invisible.

    Without this the assistant says it has no data on a metric the database holds, which
    is honest about its context and wrong about the dataset.
    """
    import numpy as np

    from footyvision.rag.store import Hit, VectorStore

    store = VectorStore(
        [1, 2],
        ["Anchor", "Runner"],
        ["Anchor profile", "Runner profile"],
        np.eye(2, 4, dtype=np.float32),
        attrs={"metrics": [{"tackles_per90": 3.41}, {"tackles_per90": 0.87}]},
    )
    hits = [Hit(1, "Anchor", "Anchor profile", 1.0, 0), Hit(2, "Runner", "Runner profile", 0.9, 1)]

    assert store.metric_notes(hits, ["tackles_per90"]) == [
        "Anchor: tackles 3.41 per 90.",
        "Runner: tackles 0.87 per 90.",
    ]
    # No metric asked, or an index built before the values were stored: say nothing rather
    # than break.
    assert store.metric_notes(hits, []) == []
    bare = VectorStore([1], ["A"], ["a"], np.eye(1, 4, dtype=np.float32))
    assert bare.metric_notes([Hit(1, "A", "a", 1.0, 0)], ["tackles_per90"]) == []


def test_prompt_lists_the_asked_for_metric_apart_from_the_profiles():
    from footyvision.rag.assistant import build_prompt

    _, user = build_prompt("most tackles?", [], None, None, ["Anchor: tackles 3.41 per 90."])

    assert "The metric asked about" in user
    assert "Anchor: tackles 3.41 per 90." in user


def test_the_prompt_forbids_reading_the_shortlist_as_a_census():
    """Asked for young players in Liga F, the assistant said the league was not in the data.

    It is: 275 players. The age filter had removed every one of them, because no woman in
    this database has a recorded date of birth, so the six retrieved were men from other
    leagues — described accurately, and used to conclude something false about the dataset.
    Disclosing the exclusion was not enough on its own.
    """
    from footyvision.rag.assistant import build_prompt
    from footyvision.rag.constraints import Constraints

    system, _ = build_prompt(
        "young players in Liga F", [], Constraints(max_age=23.0), {"age": 1154}
    )

    assert "NOT the whole database" in system
    assert "Never say a competition" in system


def test_accented_and_unaccented_spellings_are_the_same_word():
    from footyvision.rag.constraints import parse_constraints

    assert parse_constraints("médios jovens") == parse_constraints("medios jovens")
    assert parse_constraints("médios jovens").position_group == "MID"


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


def test_players_dropped_for_a_missing_attribute_are_counted_separately():
    """Failing a filter and being unfilterable are not the same fact.

    Date of birth and preferred foot come from a men's football export, so a foot or age
    requirement removes every player in the women's competitions without their ever being
    compared against it. That is a limit of the data, and the answer has to be able to
    say so rather than presenting a shortlist drawn from half the pool as the whole one.
    """
    from footyvision.rag.constraints import Constraints

    store = _attr_store()

    assert store.unknown_dropped(Constraints(foot="left")) == {"foot": 1}
    # Every age is recorded here, so an age filter hides nobody.
    assert store.unknown_dropped(Constraints(max_age=23.0)) == {}
    # A constraint that was not asked for reports nothing, even though feet are missing.
    assert store.unknown_dropped(Constraints(position_group="FWD")) == {}


def test_prompt_discloses_the_players_it_could_not_check():
    from footyvision.rag.assistant import build_prompt
    from footyvision.rag.constraints import Constraints

    system, _ = build_prompt("a left-footed winger", [], Constraints(foot="left"), {"foot": 989})

    assert "989 players have no recorded foot" in system


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


# --- the index lives in Postgres --------------------------------------------------------


def _seeded_store(db_session):
    import numpy as np

    from footyvision.rag.store import VectorStore

    return VectorStore(
        [1, 2],
        ["Alpha Striker", "Bravo Striker"],
        ["alpha profile", "bravo profile"],
        np.eye(2, 4, dtype=np.float32),
        attrs={"foot": ["left", None], "age": [21.0, np.nan], "position_group": ["FWD", "FWD"]},
    )


def test_index_round_trips_through_the_database(db_session):
    """The file is on an ephemeral disk in production; the database is not."""
    from footyvision.rag.store import VectorStore

    saved = _seeded_store(db_session).save_db(db_session, "test-embedder")
    loaded, built_with = VectorStore.load_db(db_session)

    assert saved == 2
    assert built_with == "test-embedder"
    assert list(loaded.ids) == [1, 2]
    assert loaded.vectors.shape == (2, 4)


def test_stored_index_keeps_the_filter_attributes(db_session):
    from footyvision.rag.constraints import Constraints
    from footyvision.rag.store import VectorStore

    _seeded_store(db_session).save_db(db_session, "test-embedder")
    loaded, _ = VectorStore.load_db(db_session)

    # The unknown foot must still fail the filter after a round trip, not pass it.
    assert list(loaded.ids[loaded.matching(Constraints(foot="left"))]) == [1]


def test_saving_the_index_replaces_the_previous_one(db_session):
    """A rebuild must not leave half of an older, differently-sized index behind."""
    from footyvision.rag.store import VectorStore

    _seeded_store(db_session).save_db(db_session, "old-embedder")
    VectorStore([9], ["Only One"], ["p"], __import__("numpy").eye(1, 4, dtype="float32")).save_db(
        db_session, "new-embedder"
    )

    loaded, built_with = VectorStore.load_db(db_session)

    assert list(loaded.ids) == [9]
    assert built_with == "new-embedder"


def test_load_db_returns_none_when_nothing_is_stored(db_session):
    from footyvision.rag.store import VectorStore

    assert VectorStore.load_db(db_session) is None

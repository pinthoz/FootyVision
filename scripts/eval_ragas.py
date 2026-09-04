"""Score the scouting assistant with RAGAS.

The evaluation already in this repo (`eval_retrieval_constraints.py`) measures retrieval
against ground truth we can compute: a question asking for left-footed wingers either got
left-footed wingers back or it did not. That says nothing about the half of the pipeline
that follows — whether the written answer is actually supported by the profiles retrieved,
or whether it quietly invents a number.

That is what RAGAS judges, with an LLM reading the answer against its sources. Three
metrics, none of which needs a hand-written reference answer:

  faithfulness        every claim in the answer traced back to a retrieved profile. The
                      one that matters most here: the system prompt promises never to
                      invent players or statistics, and this is the only thing that checks.
  context precision   whether the profiles retrieved were relevant to the question, judged
                      without a reference, so it measures retrieval rather than agreement
                      with an answer key someone wrote.
  answer relevancy    whether the answer addresses what was asked, rather than something
                      adjacent it happened to retrieve.

Read the numbers knowing who produced them. The judge is Gemini and so is the assistant,
and a model scoring its own family's output is a known bias in LLM-as-judge evaluation —
it tends to be generous. The scores are useful as a *relative* signal across runs and for
catching outright ungrounded answers; they are not an absolute grade.

Treat context precision with particular suspicion on a small judge model. In the first
run it returned exactly 0.00 for seven of eleven questions, including one whose answer
quotes three players and six figures straight out of the retrieved profiles — while the
deterministic evaluation puts role precision at 0.969 on the same retriever. A metric
disagreeing that hard with a measurement of the same thing is reporting on itself, not on
the pipeline. Faithfulness and answer relevancy held up under inspection; this one did
not, and it should be re-run against a stronger judge before anyone believes it.

Budget before you start. Google meters the free tier *per model per day*, and for
gemini-2.5-flash that allowance is twenty requests — a single run of this script needs
more than forty, so it will stop partway through with a 429 that looks like a per-minute
limit and is not. Both stages resume from the cache, so a stopped run is not a lost one,
but the practical answers are to point `--judge-model` at a model with its own untouched
allowance, or to run everything against a local endpoint where there is no ceiling.

Usage:
    python scripts/eval_ragas.py --generate     # run the assistant, cache the answers
    python scripts/eval_ragas.py --score        # score the cached run, no LLM generation
    python scripts/eval_ragas.py                # both, resuming whatever is cached

    # A fresh daily allowance, without changing what production is configured to use:
    python scripts/eval_ragas.py --answer-model gemini-3.1-flash-lite \
                                 --judge-model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footyvision.config import get_settings  # noqa: E402

CACHE = Path(".eval_cache/ragas/run.json")


@dataclass(frozen=True)
class Question:
    text: str
    kind: str


# Bilingual on purpose, and weighted towards what the assistant is actually asked. The
# last two have no answer in the data: the honest reply is to say so, and a model that
# invents one instead should lose faithfulness rather than pass unnoticed.
QUESTIONS: tuple[Question, ...] = (
    Question("Who are the best left-footed wingers?", "constraint"),
    Question("Quais sao os melhores extremos canhotos?", "constraint"),
    Question("Find me a young centre back who is good on the ball.", "constraint"),
    Question("Quais os medios experientes com mais desarmes?", "constraint"),
    Question("Which Brazilian forwards stand out?", "nationality"),
    Question("Que jogadoras brasileiras se destacam?", "nationality"),
    Question("Compare Neymar and Luis Suarez on dribbling.", "comparison"),
    Question("Who has the highest xG per 90 among centre forwards?", "metric"),
    Question("Quais os guarda-redes que mais recuperam bolas?", "metric"),
    Question("Which players win the most aerial duels?", "unanswerable-metric"),
    Question("Quao bom e o Otavio do FC Porto?", "unanswerable-player"),
)


def _is_rate_limit(error: Exception) -> bool:
    return "429" in str(error) or "rate limit" in str(error).lower()


def with_backoff(call, tries: int = 5, base: float = 30.0):
    """Retry through a rate limit, and give up on anything else immediately.

    The free Gemini tier allows a handful of requests a minute, and a run of this
    length will meet that ceiling. Waiting is the correct response to a 429 and only to
    a 429 — retrying a malformed request just spends the quota faster.
    """
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:
            if not _is_rate_limit(error) or attempt == tries - 1:
                raise
            wait = base * (attempt + 1)
            print(f"      rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def save(rows: list[dict], cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def generate(
    limit: int | None, pause: float, cache: Path, rows: list[dict], model: str | None
) -> list[dict]:
    """Run the real assistant over the questions not already answered.

    Saved after every question, and existing answers are kept. A rate limit two thirds of
    the way through a run used to discard the whole thing; now the same command picks up
    where it stopped.
    """
    from footyvision.db.base import SessionLocal
    from footyvision.llm.client import LLMClient
    from footyvision.rag.assistant import ScoutAssistant
    from footyvision.rag.service import get_store

    questions = QUESTIONS[:limit] if limit else QUESTIONS
    done = {row["question"] for row in rows}
    todo = [q for q in questions if q.text not in done]
    if not todo:
        print(f"  all {len(questions)} answers already cached")
        return rows

    with SessionLocal() as session:
        store = get_store(session)
        assistant = ScoutAssistant(store, client=LLMClient(cloud_model=model))
        for i, question in enumerate(todo, 1):
            try:
                result = with_backoff(lambda q=question: assistant.answer(q.text))
            except Exception as error:
                print(f"\n  stopped at {question.text!r}: {error}")
                print(f"  {len(rows)} answers kept; re-run to continue.")
                break
            rows.append(
                {
                    "question": question.text,
                    "kind": question.kind,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "filters": result.get("filters"),
                    "not_considered": result.get("not_considered"),
                }
            )
            save(rows, cache)
            print(f"  [{i}/{len(todo)}] {question.text[:56]:58s} "
                  f"{len(result['contexts'])} contexts", flush=True)
            time.sleep(pause)
    return rows


def _judge(model: str | None):
    """A RAGAS LLM and embedder backed by the same cloud provider the app uses."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    settings = get_settings()
    if not settings.active_cloud_api_key:
        raise SystemExit("No cloud API key configured; RAGAS needs a judge model.")
    # Gemini speaks the OpenAI protocol at this endpoint, which is what RAGAS expects.
    # Async, not sync: every metric's synchronous `score()` is a wrapper that calls the
    # async path underneath, and a sync client makes it fail rather than fall back.
    client = AsyncOpenAI(
        base_url=settings.cloud_llm_base_url, api_key=settings.active_cloud_api_key
    )
    llm = llm_factory(
        model or settings.cloud_llm_model,
        provider="openai",
        client=client,
        # Gemini 2.5 Flash reasons before it answers, and that reasoning is billed against
        # the same budget as the JSON it has to emit. At the default the structured output
        # is cut off mid-object and instructor raises rather than returning a partial score.
        max_tokens=4096,
        # A judge that changes its mind between runs cannot be compared against itself.
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(client=client, model=settings.cloud_llm_embed_model)
    return llm, embeddings


def contexts_for(row: dict) -> list[str]:
    """Everything the answer was legitimately grounded in, not only the profiles.

    The assistant is also told which filter it applied and how many players could not be
    checked against it, and it is instructed to say so — "1161 players have no recorded
    foot, so they were left out of the search". That sentence is true and comes from the
    pipeline, but it appears in no profile, so a faithfulness judge shown only the
    profiles marks it as invented. Scoring it that way would push the assistant towards
    dropping the disclosure, which is the opposite of what it should do. The fix is to
    show the judge the same grounding the model had.
    """
    notes: list[str] = []
    if row.get("filters"):
        notes.append(f"Retrieval note: the pool was filtered to players who are {row['filters']}.")
    for field, count in (row.get("not_considered") or {}).items():
        notes.append(
            f"Retrieval note: {count} players have no recorded {field}, so they could not be "
            f"checked against this requirement and were left out of the search."
        )
    return list(row["contexts"]) + notes


def score(
    rows: list[dict], pause: float, cache: Path, model: str | None
) -> dict[str, list[float]]:
    """Score every cached answer that has not been scored yet, saving as it goes."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    llm, embeddings = _judge(model)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }

    pending = [row for row in rows if "scores" not in row]
    for i, row in enumerate(pending, 1):
        question, answer = row["question"], row["answer"]
        contexts = contexts_for(row)
        try:
            values = {}
            for name, metric in metrics.items():
                # Answer relevancy asks whether the response fits the question, so it is
                # the one metric that takes no contexts.
                kwargs = {"user_input": question, "response": answer}
                if name != "answer_relevancy":
                    kwargs["retrieved_contexts"] = contexts
                values[name] = float(with_backoff(lambda m=metric, k=kwargs: m.score(**k)).value)
                time.sleep(pause)
        except Exception as error:
            print(f"\n  stopped scoring at {question!r}: {error}")
            print(f"  {len(rows) - len(pending) + i - 1} rows scored; re-run to continue.")
            break
        row["scores"] = values
        save(rows, cache)
        line = "  ".join(f"{n.split('_')[0]} {v:.2f}" for n, v in values.items())
        print(f"  [{i}/{len(pending)}] {row['kind']:20s} {line}", flush=True)

    scored: dict[str, list[float]] = {name: [] for name in metrics}
    for row in rows:
        for name, value in row.get("scores", {}).items():
            scored[name].append(value)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Only run the assistant.")
    parser.add_argument("--score", action="store_true", help="Only score the cached run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pause",
        type=float,
        default=15.0,
        help=(
            "Seconds between calls. The free Gemini tier allows ten requests a minute and "
            "answering one question costs two of them — the question is embedded and then "
            "the answer written — so anything below about twelve seconds hits a 429."
        ),
    )
    parser.add_argument("--cache", type=Path, default=CACHE)
    # Google meters the free tier per model per day, and for some models that budget
    # is twenty requests -- less than one run of this script. Overriding the model is
    # how you get a fresh allowance without touching what production is configured to
    # use, and pointing both at a local endpoint removes the ceiling entirely.
    parser.add_argument(
        "--answer-model", default=None, help="Model that writes the answers."
    )
    parser.add_argument(
        "--judge-model", default=None, help="Model that scores them."
    )
    args = parser.parse_args()

    do_generate = args.generate or not args.score
    do_score = args.score or not args.generate

    # Both stages resume from the cache, so a rate limit costs the questions still to do
    # rather than the ones already paid for.
    rows: list[dict] = []
    if args.cache.exists():
        rows = json.loads(args.cache.read_text(encoding="utf-8"))

    if do_generate:
        print("Running the assistant over the question set...")
        rows = generate(args.limit, args.pause, args.cache, rows, args.answer_model)
        print(f"\n{len(rows)} answers cached in {args.cache}\n")

    if not do_score:
        return
    if args.limit:
        rows = rows[: args.limit]

    print("Scoring with RAGAS...")
    scores = score(rows, args.pause, args.cache, args.judge_model)

    report(rows, scores)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def report(rows: list[dict], scores: dict[str, list[float]]) -> None:
    """Print the means, then the two things a bare mean gets wrong here.

    Answer relevancy detects a refusal and scores it zero on purpose, so the questions
    written specifically to have no answer drag the average down for behaving correctly.
    Averaging them in would reward an assistant that made something up.
    """
    print("\n" + "-" * 58)
    for name, values in scores.items():
        print(f"  {name:20s} {_mean(values):.3f}   (n={len(values)})")

    scored = [r for r in rows if "scores" in r]
    answerable = [r for r in scored if not r["kind"].startswith("unanswerable")]
    refusals = [r for r in scored if r["kind"].startswith("unanswerable")]

    if refusals:
        print(
            f"\n  On the {len(refusals)} questions the data cannot answer, faithfulness is "
            f"{_mean([r['scores']['faithfulness'] for r in refusals]):.3f}\n"
            "  — the assistant declined rather than inventing. Their answer relevancy is 0 "
            "by\n  design: the metric scores a refusal as irrelevant, which is right for a "
            "question\n  that had an answer and wrong for one that never did."
        )
    if answerable:
        print(
            f"\n  Answer relevancy on the {len(answerable)} answerable questions only: "
            f"{_mean([r['scores']['answer_relevancy'] for r in answerable]):.3f}"
        )

    print("\n  per question:")
    for row in scored:
        s = row["scores"]
        print(
            f"    {row['kind']:20s} f {s['faithfulness']:.2f}  cp {s['context_precision']:.2f}"
            f"  ar {s['answer_relevancy']:.2f}   {row['question'][:44]}"
        )
    print("-" * 58)


if __name__ == "__main__":
    main()

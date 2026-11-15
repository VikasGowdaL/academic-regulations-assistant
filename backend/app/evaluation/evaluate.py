
"""Evaluate the RAG chain using the same retrieved chunks used for generation.

Run from backend/:

python -m app.evaluation.evaluate \
    --dataset app/evaluation/questions.jsonl \
    --store faiss \
    --k 2 \
    --output app/evaluation/faiss_results.json
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_cohere import ChatCohere

from app.rag import chain
from app.rag import multi_query_retriever


load_dotenv()

REFUSAL = "i could not find sufficient information"

CATEGORIES = {
    "core_rules": 15,
    "calendar_curriculum": 10,
    "multi_part": 8,
    "follow_up": 7,
    "out_of_scope": 10,
}


# ---------------------------------------------------------
# RATE LIMIT HANDLING
# ---------------------------------------------------------

def is_rate_limit_error(exc):
    """Detect Cohere HTTP 429 rate-limit errors."""
    status = getattr(exc, "status_code", None)

    if status is None:
        status = getattr(
            getattr(exc, "response", None),
            "status_code",
            None,
        )

    message = str(exc).lower()

    return (
        status == 429
        or "status_code: 429" in message
        or "status code: 429" in message
        or "429 too many requests" in message
        or "rate limit" in message
        or "calls / minute" in message
    )


def call_with_retry(
    func,
    *args,
    max_retries=6,
    initial_delay=35,
    max_delay=120,
    **kwargs,
):
    """Retry Cohere rate-limit errors with exponential backoff."""

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)

        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if attempt >= max_retries:
                raise

            delay = min(
                initial_delay * (2 ** attempt),
                max_delay,
            )

            delay += random.uniform(0, 3)

            print(
                f"\nCohere rate limit reached (429). "
                f"Retry {attempt + 1}/{max_retries} "
                f"in {delay:.1f} seconds...",
                flush=True,
            )

            time.sleep(delay)


# ---------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------

def load_dataset(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 50 or len({
        r["id"] for r in rows
    }) != 50:
        raise ValueError(
            "Expected 50 questions with unique IDs"
        )

    counts = defaultdict(int)
    sessions_seen = set()

    for row in rows:
        for field in (
            "id",
            "category",
            "question",
            "out_of_scope",
            "reference_answer",
            "relevant_sources",
        ):
            if field not in row:
                raise ValueError(
                    f"{row.get('id', '?')}: missing {field}"
                )

        counts[row["category"]] += 1

        if row["out_of_scope"] != (
            row["category"] == "out_of_scope"
        ):
            raise ValueError(
                f"{row['id']}: incorrect scope label"
            )

        if (
            not row["out_of_scope"]
            and not row["relevant_sources"]
        ):
            raise ValueError(
                f"{row['id']}: missing gold sources"
            )

        if (
            row.get("is_follow_up")
            and row.get("session_id")
            not in sessions_seen
        ):
            raise ValueError(
                f"{row['id']}: follow-up lacks "
                "an earlier session anchor"
            )

        if row.get("session_id"):
            sessions_seen.add(row["session_id"])

    if dict(counts) != CATEGORIES:
        raise ValueError(
            f"Incorrect category distribution: {dict(counts)}"
        )

    return rows


# ---------------------------------------------------------
# LLM-AS-JUDGE
# ---------------------------------------------------------

def judge(
    llm,
    question,
    expected,
    answer,
    contexts,
    history,
):
    prompt = f"""You are evaluating an academic regulations RAG answer.

Use the reference answer and retrieved context only.

Return a single JSON object containing:
- correctness: integer 0 or 1
- faithfulness: integer 0 or 1
- reasons: short explanations

Correctness = 1 only when all essential reference
facts are correctly stated in the generated answer.

Faithfulness = 1 only when every substantive claim
in the generated answer is supported by retrieved context.

Conversation history can resolve references in the
question but is NOT evidence for policy facts.

PREVIOUS QUESTIONS AND ANSWERS:
{history[:3500]}

QUESTION:
{question}

REFERENCE ANSWER:
{expected}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXT:
{contexts[:18000]}

Return valid JSON only.
"""

    response = call_with_retry(
        llm.invoke,
        prompt,
    )

    raw = response.content

    if not isinstance(raw, str):
        raw = str(raw)

    match = re.search(
        r"\{.*\}",
        raw,
        re.S,
    )

    if not match:
        raise ValueError(
            "Judge response was not JSON"
        )

    result = json.loads(match.group())

    for key in ("correctness", "faithfulness"):
        if (
            type(result.get(key)) is not int
            or result[key] not in (0, 1)
        ):
            raise ValueError(
                f"Invalid judge {key}: "
                f"{result.get(key)}"
            )

    return result


# ---------------------------------------------------------
# METRIC HELPER
# ---------------------------------------------------------

def mean(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    return (
        round(sum(values) / len(values), 4)
        if values
        else None
    )


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def run(args):
    rows = load_dataset(args.dataset)

    if not os.getenv("COHERE_API_KEY"):
        raise EnvironmentError(
            "COHERE_API_KEY missing: "
            "set it in backend/.env"
        )

    judge_llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    histories = defaultdict(list)

    # Clear conversation memory between evaluation runs.
    chain.session_memories.clear()

    for index, row in enumerate(rows):

        session = "eval_" + row.get(
            "session_id",
            row["id"],
        )

        previous_history = "\n".join(
            histories[session]
        )

        captured = {
            "docs": None,
            "queries": [],
        }

        original_retrieve = (
            chain.retrieve_for_multiple_queries
        )

        original_planner = (
            multi_query_retriever.plan_queries
        )

        def record_retrieve(*a, **kw):
            docs = original_retrieve(*a, **kw)

            captured["docs"] = docs

            return docs

        def record_planner(*a, **kw):
            queries = original_planner(*a, **kw)

            captured["queries"].extend(queries)

            return queries

        try:
            # Capture the exact documents and planned queries
            # used by the actual RAG generation call.

            chain.retrieve_for_multiple_queries = (
                record_retrieve
            )

            multi_query_retriever.plan_queries = (
                record_planner
            )

            # Retry the entire question if a Cohere
            # rate-limit error occurs.
            #
            # Clear captured data before each attempt.

            def generate():
                captured["docs"] = None
                captured["queries"] = []

                return chain.generate_answer(
                    row["question"],
                    store_type=args.store,
                    k=args.k,
                    session_id=session,
                )

            response = call_with_retry(generate)

            docs = captured["docs"]

            if docs is None:
                raise RuntimeError(
                    "Could not capture retrieval "
                    "used by generate_answer"
                )

            answer = response["answer"]

            refused = (
                REFUSAL in answer.lower()
            )

            retrieved = {
                (
                    doc.metadata.get("source"),
                    doc.metadata.get("page_number"),
                )
                for doc in docs
            }

            expected = {
                (
                    source["source"],
                    source["page"],
                )
                for source in row["relevant_sources"]
            }

            recall = (
                len(expected & retrieved) / len(expected)
                if expected
                else None
            )

            context = "\n\n".join(
                (
                    f"Source: {doc.metadata.get('source')} "
                    f"Page: {doc.metadata.get('page_number')}\n"
                    f"{doc.page_content}"
                )
                for doc in docs
            )

            scores = None

            if (
                not row["out_of_scope"]
                and not refused
            ):
                scores = judge(
                    judge_llm,
                    row["question"],
                    row["reference_answer"],
                    answer,
                    context,
                    previous_history,
                )

            result = {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "session_id": session,
                "is_follow_up": row.get(
                    "is_follow_up",
                    False,
                ),
                "out_of_scope": row["out_of_scope"],
                "answer": answer,
                "refused": refused,
                "refusal_decision_correct": (
                    refused == row["out_of_scope"]
                ),
                "retrieval_recall_at_k_page_level": recall,
                "correctness": (
                    (
                        0
                        if refused
                        else scores["correctness"]
                    )
                    if not row["out_of_scope"]
                    else None
                ),
                "faithfulness": (
                    scores["faithfulness"]
                    if scores
                    else None
                ),
                "planned_queries": captured["queries"],
                "planned_query_count": len(
                    captured["queries"]
                ),
                "retrieved_sources": [
                    {
                        "source": name,
                        "page": page,
                    }
                    for name, page in sorted(retrieved)
                ],
                "judge_details": scores,
            }

            histories[session].extend([
                "User: " + row["question"],
                "Assistant: " + answer,
            ])

        except Exception as exc:
            result = {
                "id": row["id"],
                "category": row["category"],
                "error": str(exc),
            }

        finally:
            chain.retrieve_for_multiple_queries = (
                original_retrieve
            )

            multi_query_retriever.plan_queries = (
                original_planner
            )

        results.append(result)

        # Save after every question to avoid losing
        # already completed evaluation results.

        output.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        status = (
            "ERROR"
            if "error" in result
            else "OK"
        )

        print(
            f"{index + 1}/{len(rows)} "
            f"{row['id']}: {status}",
            flush=True,
        )

        # Delay between questions to reduce the
        # likelihood of exceeding the Cohere trial
        # key's 20 API calls/minute limit.
        #
        # This is not a strict per-request rate limiter.

        if index < len(rows) - 1:
            print(
                f"Waiting {args.delay} seconds...",
                flush=True,
            )

            time.sleep(args.delay)

    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------

    valid = [
        row
        for row in results
        if "error" not in row
    ]

    inscope = [
        row
        for row in valid
        if not row["out_of_scope"]
    ]

    oos = [
        row
        for row in valid
        if row["out_of_scope"]
    ]

    categories = {
        category: {
            "total": sum(
                row["category"] == category
                for row in rows
            ),
            "completed": sum(
                row["category"] == category
                for row in valid
            ),
            "correctness": mean(
                [
                    row
                    for row in inscope
                    if row["category"] == category
                ],
                "correctness",
            ),
            "page_recall": mean(
                [
                    row
                    for row in inscope
                    if row["category"] == category
                ],
                "retrieval_recall_at_k_page_level",
            ),
        }
        for category in CATEGORIES
    }

    summary = {
        "total": len(rows),
        "completed": len(valid),
        "errors": len(rows) - len(valid),
        "vector_store": args.store,
        "k_per_planned_query": args.k,
        "page_level_recall_mean_in_scope": mean(
            inscope,
            "retrieval_recall_at_k_page_level",
        ),
        "correctness_mean_in_scope_refusals_zero": mean(
            inscope,
            "correctness",
        ),
        "faithfulness_mean_answered_in_scope": mean(
            inscope,
            "faithfulness",
        ),
        "out_of_scope_refusal_accuracy": mean(
            oos,
            "refused",
        ),
        "overall_refusal_decision_accuracy": mean(
            valid,
            "refusal_decision_correct",
        ),
        "multi_part_with_two_or_more_planned_queries": sum(
            row["planned_query_count"] >= 2
            for row in valid
            if row["category"] == "multi_part"
        ),
        "multi_part_completed": sum(
            row["category"] == "multi_part"
            for row in valid
        ),
        "follow_up_correctness": mean(
            [
                row
                for row in inscope
                if row["category"] == "follow_up"
            ],
            "correctness",
        ),
        "by_category": categories,
        "limitations": [
            "Recall is source-page recall, not chunk-level recall.",
            "k is applied to each planned sub-query, not a global top-k.",
            "Follow-up sessions exercise memory but do not isolate its causal benefit; use a memory-off ablation for that claim.",
            "The 1.0 retrieval threshold is experimental and may need calibration separately for Chroma.",
            "LLM judge scores are not ground truth; manually audit a sample.",
            "Incomplete runs should not be reported as full evaluation.",
        ],
    }

    summary_path = output.with_name(
        output.stem + "_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEVALUATION COMPLETE\n")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


# ---------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="app/evaluation/questions.jsonl",
    )

    parser.add_argument(
        "--store",
        choices=("faiss", "chroma"),
        default="faiss",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output",
        default="app/evaluation/faiss_results.json",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=25.0,
        help="Delay in seconds between evaluation questions",
    )

    run(parser.parse_args())

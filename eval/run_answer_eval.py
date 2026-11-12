"""Evaluate the full answer pipeline: correctness, citations, and refusal."""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.answer import parse_citations
from core.generator import answer_question
from eval.answer_judge import (
    choose_judge_model,
    fallback_judge_model,
    judge_answer,
)
from eval.answer_metrics import citation_scores, token_f1
from eval.run_retrieval_eval import (
    build_index,
    collect_corpus_files,
    display_path,
    load_dataset,
    relevant_ids_for,
)

REFUSAL_PHRASES = (
    "无法回答",
    "没有足够信息",
    "信息不足",
    "未找到",
    "无法确定",
    "知识库为空",
)


def strip_details(text):
    """Remove collapsible thinking blocks and remaining HTML tags."""
    text = re.sub(r"<details>.*?</details>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()


def clean_answer(text):
    """Remove thinking/HTML and [S1] labels before lexical scoring."""
    text = strip_details(text)
    text = re.sub(r"\[S\d+\]", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer correctness and citation accuracy"
    )
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "eval" / "golden_qa.jsonl"
    )
    parser.add_argument("--corpus", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model-choice", default="ollama")
    parser.add_argument(
        "--judge-model",
        choices=["auto", "deepseek", "glm"],
        default=None,
        help="Enable LLM correctness/faithfulness judging",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    corpus_files = collect_corpus_files(args.corpus)
    if not corpus_files:
        raise SystemExit("no corpus files found; pass --corpus")
    entries = load_dataset(args.dataset)[: args.limit]
    chunks, ids, documents = build_index(corpus_files)
    judge_model = (
        choose_judge_model(args.model_choice, args.judge_model)
        if args.judge_model
        else None
    )
    fallback_model = (
        fallback_judge_model(judge_model) if judge_model else None
    )

    print("Documents:")
    for document in documents:
        print(f"- {document['source']}: {document['chunks']} chunks")
    print(
        f"Corpus files: {len(corpus_files)} | Chunks: {len(chunks)} | "
        f"Queries: {len(entries)} | model={args.model_choice} | "
        f"judge={judge_model or 'off'} | fallback={fallback_model or 'off'}"
    )

    rows = []
    for entry in entries:
        result = answer_question(
            entry["question"],
            enable_web_search=False,
            model_choice=args.model_choice,
        )
        relevant_ids = relevant_ids_for(
            entry["relevant_snippets"], chunks, ids
        )
        visible_answer = strip_details(result.answer)
        cited = parse_citations(visible_answer)
        sources_by_id = {source.id: source for source in result.sources}
        precision, recall = citation_scores(
            cited, sources_by_id, relevant_ids
        )
        cleaned = clean_answer(result.answer)
        f1 = token_f1(cleaned, entry["answer"])
        refused = any(phrase in cleaned for phrase in REFUSAL_PHRASES)

        cited_chunk_ids = sorted(
            {
                sources_by_id[source_id].chunk_id
                for source_id in cited
                if source_id in sources_by_id
                and sources_by_id[source_id].chunk_id
            }
        )
        retrieved_chunk_ids = sorted(
            {
                source.chunk_id
                for source in result.sources
                if source.chunk_id
            }
        )
        relevant_chunk_ids = sorted(relevant_ids)
        retrieval_hit = bool(set(retrieved_chunk_ids) & relevant_ids)

        judge_result = None
        if judge_model:
            judge_result = judge_answer(
                entry["question"],
                entry["answer"],
                visible_answer,
                result.sources,
                judge_model=judge_model,
                fallback_model=fallback_model,
            )

        row = {
            "id": entry["id"],
            "question": entry["question"],
            "token_f1": f1,
            "citation_precision": precision,
            "citation_recall": recall,
            "cited_ids": sorted(cited),
            "cited_chunk_ids": cited_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "relevant_chunk_ids": relevant_chunk_ids,
            "relevant_chunks": len(relevant_ids),
            "retrieval_hit": retrieval_hit,
            "citation_adherence": bool(cited),
            "answer_chars": len(cleaned),
            "refused": refused,
            "answer_preview": cleaned[:300],
        }
        if judge_result is not None:
            row.update(
                {
                    "judge_model": judge_result.get("judge_model"),
                    "correct": judge_result.get("correct"),
                    "correctness_score": judge_result.get("correctness_score"),
                    "faithful": judge_result.get("faithful"),
                    "unsupported_claims": judge_result.get(
                        "unsupported_claims", []
                    ),
                    "citation_support": judge_result.get(
                        "citation_support", []
                    ),
                    "judge_reason": judge_result.get("reason", ""),
                    "judge_error": judge_result.get("judge_error"),
                    "judge_fallback_used": judge_result.get(
                        "judge_fallback_used", False
                    ),
                    "judge_primary_error": judge_result.get(
                        "judge_primary_error"
                    ),
                }
            )
        rows.append(row)
        print(
            f"- {row['id']}: f1={f1:.3f} "
            f"citation_p={precision:.3f} citation_r={recall:.3f} "
            f"retrieval_hit={retrieval_hit} "
            f"citation={row['citation_adherence']} "
            f"chars={row['answer_chars']}"
        )
        if judge_result is not None:
            print(
                f"  judge[{judge_result.get('judge_model')}]: "
                f"correct={judge_result.get('correct')} "
                f"score={judge_result.get('correctness_score')} "
                f"faithful={judge_result.get('faithful')} "
                f"fallback={judge_result.get('judge_fallback_used', False)} "
                f"error={judge_result.get('judge_error')}"
            )

    def mean(key):
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    metrics = {
        "token_f1": mean("token_f1"),
        "citation_precision": mean("citation_precision"),
        "citation_recall": mean("citation_recall"),
        "refusal_rate": (
            sum(1 for row in rows if row["refused"]) / len(rows)
            if rows
            else 0.0
        ),
        "retrieval_hit_rate": (
            sum(1 for row in rows if row["retrieval_hit"]) / len(rows)
            if rows
            else 0.0
        ),
        "citation_adherence_rate": (
            sum(1 for row in rows if row["citation_adherence"]) / len(rows)
            if rows
            else 0.0
        ),
        "mean_answer_chars": mean("answer_chars"),
    }

    if judge_model:
        judged = [row for row in rows if row.get("judge_error") is None]
        errors = [row for row in rows if row.get("judge_error")]
        scores = [
            float(row["correctness_score"])
            for row in judged
            if row.get("correctness_score") is not None
        ]
        citation_entries = [
            item
            for row in judged
            for item in row.get("citation_support", [])
            if isinstance(item, dict)
        ]
        supported = [
            item for item in citation_entries
            if item.get("supported") is True
        ]
        metrics.update(
            {
                "correctness_rate": (
                    sum(1 for row in judged if row.get("correct") is True)
                    / len(judged)
                    if judged
                    else 0.0
                ),
                "mean_correctness_score": (
                    sum(scores) / len(scores) if scores else 0.0
                ),
                "faithfulness_rate": (
                    sum(1 for row in judged if row.get("faithful") is True)
                    / len(judged)
                    if judged
                    else 0.0
                ),
                "unsupported_claim_count": sum(
                    len(row.get("unsupported_claims") or [])
                    for row in judged
                ),
                "citation_support_rate": (
                    len(supported) / len(citation_entries)
                    if citation_entries
                    else 0.0
                ),
                "judge_error_rate": (
                    len(errors) / len(rows) if rows else 0.0
                ),
                "judge_fallback_rate": (
                    sum(
                        1 for row in rows
                        if row.get("judge_fallback_used")
                    ) / len(rows)
                    if rows
                    else 0.0
                ),
            }
        )

    print("\nSummary:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    report = {
        "corpus_files": [display_path(path) for path in corpus_files],
        "dataset": display_path(args.dataset),
        "model_choice": args.model_choice,
        "judge_requested": args.judge_model,
        "judge_model": judge_model,
        "judge_fallback_model": fallback_model,
        "limit": args.limit,
        "metrics": metrics,
        "per_query": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_out}")


if __name__ == "__main__":
    main()

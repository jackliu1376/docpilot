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


def clean_answer(text):
    """Remove HTML and [S1] labels before lexical scoring."""
    text = re.sub(r"<[^>]+>", " ", text)
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
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    corpus_files = collect_corpus_files(args.corpus)
    if not corpus_files:
        raise SystemExit("no corpus files found; pass --corpus")
    entries = load_dataset(args.dataset)[: args.limit]
    chunks, ids, documents = build_index(corpus_files)

    print("Documents:")
    for document in documents:
        print(f"- {document['source']}: {document['chunks']} chunks")
    print(
        f"Corpus files: {len(corpus_files)} | Chunks: {len(chunks)} | "
        f"Queries: {len(entries)} | model={args.model_choice}"
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
        cited = parse_citations(result.answer)
        sources_by_id = {source.id: source for source in result.sources}
        precision, recall = citation_scores(
            cited, sources_by_id, relevant_ids
        )
        cleaned = clean_answer(result.answer)
        f1 = token_f1(cleaned, entry["answer"])
        refused = any(phrase in cleaned for phrase in REFUSAL_PHRASES)

        row = {
            "id": entry["id"],
            "question": entry["question"],
            "token_f1": f1,
            "citation_precision": precision,
            "citation_recall": recall,
            "cited_ids": sorted(cited),
            "relevant_chunks": len(relevant_ids),
            "refused": refused,
            "answer_preview": cleaned[:300],
        }
        rows.append(row)
        print(
            f"- {row['id']}: f1={f1:.3f} "
            f"citation_p={precision:.3f} citation_r={recall:.3f} "
            f"refused={refused}"
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
    }
    print("\nSummary:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    report = {
        "corpus_files": [display_path(path) for path in corpus_files],
        "dataset": display_path(args.dataset),
        "model_choice": args.model_choice,
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

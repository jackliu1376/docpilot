"""Deterministic answer-level metrics for the evaluation harness."""

import re
from collections import Counter

import jieba


def _tokens(text):
    return [
        token
        for token in jieba.lcut(text)
        if token.strip() and not re.fullmatch(r"[\W_]+", token, flags=re.UNICODE)
    ]


def token_f1(prediction, reference):
    """Token-level F1 between the generated and reference answers."""
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0

    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def citation_scores(cited_ids, sources_by_id, relevant_chunk_ids):
    """Return (citation precision, citation recall) for cited source ids."""
    cited_chunks = {
        sources_by_id[source_id].chunk_id
        for source_id in cited_ids
        if source_id in sources_by_id
        and sources_by_id[source_id].chunk_id
    }
    relevant = set(relevant_chunk_ids)
    relevant_cited = cited_chunks & relevant

    precision = (
        len(relevant_cited) / len(cited_chunks) if cited_chunks else 0.0
    )
    recall = len(relevant_cited) / len(relevant) if relevant else 0.0
    return precision, recall

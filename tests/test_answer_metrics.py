from core.answer import Source
from eval.answer_metrics import citation_scores, token_f1
from eval.run_answer_eval import clean_answer, strip_details


def test_token_f1_identical_answers_score_one():
    assert token_f1("数据库 系统", "数据库 系统") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("数据库 系统", "数据库 设计") == 0.5


def test_citation_scores_precision_and_recall():
    sources = {
        "S1": Source(id="S1", kind="local", snippet="a", chunk_id="a"),
        "S2": Source(id="S2", kind="local", snippet="b", chunk_id="b"),
    }

    precision, recall = citation_scores({"S1", "S2"}, sources, {"a"})

    assert precision == 0.5
    assert recall == 1.0


def test_citation_scores_without_citations():
    assert citation_scores(set(), {}, {"a"}) == (0.0, 0.0)


def test_strip_details_removes_thinking_but_keeps_citations():
    text = "答案 [S1]\n<details><summary>思考</summary>推理 [S9]</details>"

    assert strip_details(text).strip() == "答案 [S1]"


def test_clean_answer_removes_details_and_citations():
    text = "答案 [S1]\n<details><summary>思考</summary>推理 [S9]</details>"

    assert clean_answer(text).strip() == "答案"

from config import RERANK_MODEL_NAME

import rag_demo


def test_system_models_info_uses_configured_reranker():
    info = rag_demo.get_system_models_info()

    assert RERANK_MODEL_NAME in info["重排序模型"]

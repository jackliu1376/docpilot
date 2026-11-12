"""
联网搜索 —— 通过 Tavily（默认）或 SerpAPI 获取实时网络信息

学习要点：
- RAG 的“R”不限于本地文档，也可以从网络获取实时信息
- 网络搜索结果不进入 FAISS/BM25 索引，仅作为文本上下文提供给 LLM
- Tavily 和 SerpAPI 都通过 SEARCH_PROVIDER 切换
"""

import logging
import requests

from config import (
    SERPAPI_KEY,
    SEARCH_ENGINE,
    SEARCH_PROVIDER,
    TAVILY_API_KEY,
    TAVILY_SEARCH_URL,
    TAVILY_SEARCH_DEPTH,
    is_configured_api_key,
)


def check_serpapi_key():
    """检查是否配置了有效的 SERPAPI_KEY（旧实现）。"""
    return is_configured_api_key(SERPAPI_KEY)


def check_search_key():
    """按 SEARCH_PROVIDER 检查当前联网搜索是否已配置。"""
    if SEARCH_PROVIDER == "tavily":
        return is_configured_api_key(TAVILY_API_KEY)
    return check_serpapi_key()


def tavily_search(query, num_results=5):
    """使用 Tavily Search API 获取网页结果。"""
    if not TAVILY_API_KEY:
        raise ValueError("未设置 TAVILY_API_KEY 环境变量")
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": TAVILY_SEARCH_DEPTH,
            "max_results": num_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=20)
        response.raise_for_status()
        return _parse_tavily_results(response.json())
    except Exception as e:
        logging.error(f"Tavily 搜索失败: {str(e)}")
        return []


def _parse_tavily_results(data):
    """把 Tavily 返回结果映射成统一的 title/url/snippet/timestamp。"""
    results = []
    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
                "timestamp": item.get("published_date")
                or item.get("published_at"),
            }
        )
    return results


def serpapi_search(query, num_results=5):
    """执行 SerpAPI 搜索（旧实现，保留兼容）。"""
    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量")
    try:
        params = {
            "engine": SEARCH_ENGINE, "q": query, "api_key": SERPAPI_KEY,
            "num": num_results, "hl": "zh-CN", "gl": "cn"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as e:
        logging.error(f"网络搜索失败: {str(e)}")
        return []


def _parse_serpapi_results(data):
    """解析 SerpAPI 返回的原始数据。"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"), "url": item.get("link"),
                "snippet": item.get("snippet"), "timestamp": item.get("date")
            })
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"), "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"), "source": "knowledge_graph"
        })
    return results


def search_web(query, num_results=5):
    """按 SEARCH_PROVIDER 执行网络搜索，结果不加入索引。"""
    if SEARCH_PROVIDER == "tavily":
        results = tavily_search(query, num_results)
    else:
        results = serpapi_search(query, num_results)

    if not results:
        logging.info("网络搜索没有返回结果")
    else:
        logging.info(f"网络搜索返回 {len(results)} 条结果")
    return results

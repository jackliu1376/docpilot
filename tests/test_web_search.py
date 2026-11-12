import features.web_search as web_search


def test_parse_tavily_results_maps_content_and_date():
    data = {
        "results": [
            {
                "title": "T",
                "url": "https://example.test",
                "content": "snippet",
                "published_date": "2026-11-01",
            }
        ]
    }

    assert web_search._parse_tavily_results(data) == [
        {
            "title": "T",
            "url": "https://example.test",
            "snippet": "snippet",
            "timestamp": "2026-11-01",
        }
    ]


def test_check_search_key_uses_tavily_key(monkeypatch):
    monkeypatch.setattr(web_search, "SEARCH_PROVIDER", "tavily")
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "")
    assert web_search.check_search_key() is False

    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "tvly-real")
    assert web_search.check_search_key() is True


def test_tavily_search_posts_expected_payload(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "T", "url": "u", "content": "s"}
                ]
            }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(web_search, "TAVILY_API_KEY", "tvly-real")
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    results = web_search.tavily_search("query", 3)

    assert results[0]["snippet"] == "s"
    url, kwargs = calls[0]
    assert url == web_search.TAVILY_SEARCH_URL
    assert kwargs["json"]["api_key"] == "tvly-real"
    assert kwargs["json"]["query"] == "query"
    assert kwargs["json"]["max_results"] == 3

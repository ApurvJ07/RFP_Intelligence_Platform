import pytest
from search.engine import HybridSearchEngine


@pytest.fixture(scope="module")
def engine():
    e = HybridSearchEngine()
    e.index_bid_folder("data/Bid1")
    e.index_bid_folder("data/Bid2")
    return e


def test_search_returns_expected_keys(engine):
    results = engine.search("submission deadline", bid_id="Bid1", top_k=3, mode="hybrid_rerank")
    assert 1 <= len(results) <= 3
    for r in results:
        for key in ("bid_id", "file_name", "page_number", "text"):
            assert key in r


def test_bid_filter_is_respected(engine):
    results = engine.search("affidavit", bid_id="Bid2", top_k=5, mode="hybrid")
    assert results
    assert all(r["bid_id"] == "Bid2" for r in results)


def test_exact_term_query_finds_mercury_affidavit(engine):
    results = engine.search("Mercury affidavit", bid_id="Bid2", top_k=3, mode="hybrid_rerank")
    assert "mercury" in results[0]["file_name"].lower()


def test_addendum_due_date_is_retrievable(engine):
    results = engine.search("new due date July 9 2024", bid_id="Bid1", top_k=10, mode="hybrid_rerank")
    assert any("july 9, 2024" in r["text"].lower() for r in results)


def test_all_three_modes_work(engine):
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        assert engine.search("delivery within 45 days", bid_id="Bid2", top_k=3, mode=mode)
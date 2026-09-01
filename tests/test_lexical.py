"""Tests for code tokenization and the BM25 lexical search index."""

import pytest

from ai_software_engineering_agent.lexical import BM25Index, CodeTokenizer
from ai_software_engineering_agent.models import CodeChunk, MetadataFilter


def test_code_tokenizer_splits_camel_case_and_snake_case():
    tokens = CodeTokenizer.tokenize("def calculate_YieldCurve_ytm(self): pass")
    assert "calculate" in tokens
    assert "yieldcurve" in tokens
    assert "yield" in tokens
    assert "curve" in tokens
    assert "ytm" in tokens
    assert "pass" in tokens


def test_code_tokenizer_preserves_identifiers_and_removes_stopwords():
    tokens = CodeTokenizer.tokenize("This is a function for the YieldCurveBootstrap")
    assert "function" in tokens
    assert "yieldcurvebootstrap" in tokens
    assert "yield" in tokens
    assert "curve" in tokens
    assert "bootstrap" in tokens
    assert "this" not in tokens
    assert "for" not in tokens
    assert "the" not in tokens


def test_bm25_index_indexes_and_searches_chunks():
    index = BM25Index()

    c1 = CodeChunk(
        chunk_id="chk1",
        repo_id="repo-1",
        file_path="models/yield.py",
        start_line=1,
        end_line=20,
        symbol_name="YieldCurve",
        language="python",
        content="class YieldCurve:\n    def bootstrap(self):\n        pass",
    )
    c2 = CodeChunk(
        chunk_id="chk2",
        repo_id="repo-1",
        file_path="models/bond.py",
        start_line=1,
        end_line=25,
        symbol_name="price_bond",
        language="python",
        content="def price_bond(coupon: float, ytm: float) -> float:\n    return 100.0",
    )
    c3 = CodeChunk(
        chunk_id="chk3",
        repo_id="repo-2",
        file_path="utils/math.py",
        start_line=1,
        end_line=10,
        symbol_name="discount",
        language="python",
        content="def discount(rate, t): return math.exp(-rate * t)",
    )

    index.index_chunks([c1, c2, c3])
    assert index.count() == 3

    # Query for exact symbol
    results = index.search("price_bond", limit=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "chk2"

    # Query for YieldCurve bootstrap
    results_curve = index.search("YieldCurve bootstrap", limit=2)
    assert len(results_curve) >= 1
    assert results_curve[0].chunk.chunk_id == "chk1"


def test_bm25_index_metadata_filtering():
    index = BM25Index()

    c1 = CodeChunk(
        chunk_id="chk1",
        repo_id="repo-1",
        file_path="models/yield.py",
        start_line=1,
        end_line=10,
        symbol_name="bootstrap",
        language="python",
        content="def bootstrap(): pass",
    )
    c2 = CodeChunk(
        chunk_id="chk2",
        repo_id="repo-2",
        file_path="docs/guide.md",
        start_line=1,
        end_line=15,
        symbol_name=None,
        language="markdown",
        content="How to bootstrap the yield curve in practice.",
    )

    index.index_chunks([c1, c2])

    # Filter by repo_id
    filter_repo = MetadataFilter(repo_id="repo-1")
    res_repo = index.search("bootstrap", limit=5, filter=filter_repo)
    assert len(res_repo) == 1
    assert res_repo[0].chunk.chunk_id == "chk1"

    # Filter by path pattern
    filter_path = MetadataFilter(path_patterns=("models/*.py",))
    res_path = index.search("bootstrap", limit=5, filter=filter_path)
    assert len(res_path) == 1
    assert res_path[0].chunk.file_path == "models/yield.py"

    # Filter by symbol_only
    filter_sym = MetadataFilter(symbol_only=True)
    res_sym = index.search("bootstrap", limit=5, filter=filter_sym)
    assert len(res_sym) == 1
    assert res_sym[0].chunk.symbol_name == "bootstrap"

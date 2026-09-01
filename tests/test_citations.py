"""Tests for citation extraction, line-range verification, and hallucination detection."""

import pytest

from ai_software_engineering_agent.models import Citation, CodeChunk, RetrievalResult
from ai_software_engineering_agent.rag import extract_citations, validate_citations


def test_validate_citations_marks_verified_citations():
    chunk = CodeChunk(
        chunk_id="chk1",
        repo_id="r1",
        file_path="models/curve.py",
        start_line=10,
        end_line=40,
        symbol_name="bootstrap",
        language="python",
        content="def bootstrap(): pass",
    )
    retrieved = [RetrievalResult(chunk=chunk, score=0.95)]

    # 1. Valid citation inside line range
    valid_citation = Citation(
        file_path="models/curve.py",
        start_line=15,
        end_line=25,
    )
    validated = validate_citations([valid_citation], retrieved)
    assert len(validated) == 1
    assert validated[0].is_verified is True

    # 2. Hallucinated file path not in retrieved context
    hallucinated_file = Citation(
        file_path="other/unknown.py",
        start_line=1,
        end_line=10,
    )
    validated_hallucinated = validate_citations([hallucinated_file], retrieved)
    assert len(validated_hallucinated) == 1
    assert validated_hallucinated[0].is_verified is False


def test_extract_citations_with_validation():
    chunk = CodeChunk(
        chunk_id="chk1",
        repo_id="r1",
        file_path="models/curve.py",
        start_line=1,
        end_line=30,
        symbol_name="YieldCurve",
        language="python",
        content="class YieldCurve: pass",
    )
    retrieved = [RetrievalResult(chunk=chunk, score=0.9)]

    text = "The curve is defined in [models/curve.py:5-20] and also mentions [fake/path.py:1-5]."
    citations = extract_citations(text, retrieved)

    assert len(citations) == 2
    c1 = next(c for c in citations if c.file_path == "models/curve.py")
    c2 = next(c for c in citations if c.file_path == "fake/path.py")

    assert c1.is_verified is True
    assert c2.is_verified is False

import pytest
from pydantic import ValidationError

from agents.workflow import (
    ExtractedField,
    SourceCitation,
    MultiAgentOrchestrator,
    is_missing,
)
import agents.llm as llm


def make_orchestrator():
    # the validator never touches the search engine, so None is fine here
    return MultiAgentOrchestrator(engine=None)


def test_is_missing_null_and_empty():
    assert is_missing("Term of Bid", None)
    assert is_missing("Term of Bid", "")
    assert is_missing("Term of Bid", "Not found in documents")


def test_pre_bid_meeting_none_is_a_valid_answer():
    assert not is_missing("Pre Bid Meeting", "None")
    assert is_missing("Installation", "None")


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        ExtractedField(field_name="Title", value="x", confidence=1.5)


def test_validator_passes_value_with_citation():
    f = ExtractedField(
        field_name="Bid Number", value="JA-207652",
        sources=[SourceCitation(file="rfp.pdf", page=1)], confidence=0.7,
    )
    report = make_orchestrator().validator_agent({"Bid Number": f})
    assert report.results[0].passed is True
    assert report.results[0].adjusted_confidence >= 0.9


def test_validator_fails_value_without_citation():
    f = ExtractedField(field_name="Bid Number", value="JA-207652", sources=[], confidence=0.9)
    report = make_orchestrator().validator_agent({"Bid Number": f})
    assert report.results[0].passed is False


def test_validator_accepts_not_found_field():
    f = ExtractedField(field_name="Model_no", value=None, sources=[], confidence=0.8,
                       notes="Not found in documents")
    report = make_orchestrator().validator_agent({"Model_no": f})
    assert report.results[0].passed is True


def test_validator_accepts_no_prebid_meeting():
    f = ExtractedField(field_name="Pre Bid Meeting", value="None", sources=[], confidence=0.8)
    report = make_orchestrator().validator_agent({"Pre Bid Meeting": f})
    assert report.results[0].passed is True


def test_llm_response_is_cached(tmp_path, monkeypatch):
    """The second identical call must come from disk, not from Gemini."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_try_model(model_name, prompt, schema_cls):
        calls["n"] += 1
        return schema_cls(field_name="Title", value="Test", sources=[], confidence=0.5, notes="")

    monkeypatch.setattr(llm, "_try_model", fake_try_model)

    first = llm.call_llm_structured("unit-test prompt", ExtractedField)
    second = llm.call_llm_structured("unit-test prompt", ExtractedField)

    assert calls["n"] == 1
    assert first == second
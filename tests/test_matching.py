"""Fast unit tests for evidence->chunk matching (no corpus needed)."""

from src.evaluation.matching import is_relevant, resolve_gold_pages
from src.evaluation.schema import Evidence, QAItem
from src.ingestion.schema import Chunk


def _qa(evidence_text: str) -> QAItem:
    return QAItem(
        id="x",
        question="q",
        answer="a",
        company="c",
        doc_name="DOC",
        question_type="t",
        evidence=[Evidence(text=evidence_text, doc_name="DOC", page=1)],
    )


def test_resolve_gold_pages_picks_best_overlap_page():
    page_words = {
        1: {"intro", "preamble"},
        2: {"capital", "expenditure", "1577", "cash"},
        3: {"notes", "policies"},
    }
    # Evidence text overlaps page 2 most.
    assert resolve_gold_pages(_qa("capital expenditure 1577"), page_words) == {2}


def test_resolve_gold_pages_empty_corpus():
    assert resolve_gold_pages(_qa("anything"), {}) == set()


def test_is_relevant():
    gold = {2}
    on_page = Chunk(chunk_id="DOC::5", doc_id="DOC", text="...", page=2)
    off_page = Chunk(chunk_id="DOC::9", doc_id="DOC", text="...", page=3)
    wrong_doc = Chunk(chunk_id="OTHER::0", doc_id="OTHER", text="...", page=2)

    assert is_relevant(on_page, "DOC", gold) is True
    assert is_relevant(off_page, "DOC", gold) is False
    assert is_relevant(wrong_doc, "DOC", gold) is False

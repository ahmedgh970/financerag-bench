"""Unit tests for the JSONL plumbing shared by the evaluation runners."""

import pytest

from src.evaluation.common.io import (
    append_jsonl,
    derived_path,
    done_ids,
    read_by_id,
    read_jsonl,
    select,
)

RECORDS = [{"id": "q1", "v": 1}, {"id": "q2", "v": 2}, {"id": "q3", "v": 3}]


def test_append_then_read_round_trips_and_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    for r in RECORDS:
        append_jsonl(path, r)
    path.write_text(path.read_text() + "\n")  # a trailing blank line must be ignored
    assert read_jsonl(path) == RECORDS
    assert read_by_id(path)["q2"] == {"id": "q2", "v": 2}


def test_done_ids_is_empty_before_the_first_write(tmp_path):
    path = tmp_path / "out.jsonl"
    assert done_ids(path) == set()
    append_jsonl(path, RECORDS[0])
    assert done_ids(path) == {"q1"}


def test_select_by_id_by_limit_or_all():
    assert select(RECORDS, qa_id="q2") == [RECORDS[1]]
    assert select(RECORDS, limit=2) == RECORDS[:2]
    assert select(RECORDS) == RECORDS


def test_select_fails_loudly_on_an_unknown_id():
    with pytest.raises(SystemExit):
        select(RECORDS, qa_id="nope")


def test_derived_path_names_the_output_after_the_answers_file_and_the_scorer():
    path = derived_path(
        "data/answers/run_a.jsonl", "data/judged", "judged_by_ollama_chat/llama3.1:8b"
    )
    assert str(path) == "data/judged/run_a_judged_by_ollama_chat_llama3.1:8b.jsonl"

"""Fast unit tests for the ingestion config and chunk storage (no parsing)."""

from src.ingestion.config import IngestConfig, load_config
from src.ingestion.schema import Chunk
from src.ingestion.storage import read_chunks, write_chunks


def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("parser: docling\nchunker: hybrid\nparser_params: {do_ocr: false}\n")

    cfg = load_config(str(yaml_file))

    assert isinstance(cfg, IngestConfig)
    assert cfg.parser == "docling"
    assert cfg.parser_params == {"do_ocr": False}
    assert cfg.chunker_params == {}


def test_resolved_output_path_default():
    cfg = IngestConfig(parser="docling", chunker="hybrid")
    assert cfg.resolved_output_path == "data/processed/docling_hybrid/chunks.jsonl"


def test_resolved_output_path_explicit():
    cfg = IngestConfig(parser="docling", chunker="hybrid", output_path="out/x.jsonl")
    assert cfg.resolved_output_path == "out/x.jsonl"


def test_parsed_dir_derived_from_parser():
    # A parse config has no chunker; parsed_dir depends only on the parser.
    cfg = IngestConfig(parser="unstructured")
    assert cfg.chunker is None
    assert cfg.parsed_dir == "data/processed/unstructured/parsed"


def test_chunks_jsonl_roundtrip(tmp_path):
    chunks = [
        Chunk(chunk_id="d::0", doc_id="d", text="hello", page=1, metadata={"labels": ["Title"]}),
        Chunk(chunk_id="d::1", doc_id="d", text="world", page=2),
    ]
    path = str(tmp_path / "chunks.jsonl")

    n = write_chunks(chunks, path)
    restored = list(read_chunks(path))

    assert n == 2
    assert restored == chunks  # pydantic models compare by value

"""Validate the Qdrant connection in both modes: local and server.

The local/in-memory tests always run (no Docker needed). The server test only
runs when a reachable Qdrant server is advertised via the ``QDRANT_URL`` env var
(e.g. after ``docker compose up -d qdrant``); otherwise it is skipped.
"""

import os

import pytest

from src.retrieval.qdrant_store import get_client


def test_memory_connection():
    # ":memory:" gives a working in-process client.
    client = get_client(":memory:")
    assert client.get_collections() is not None


def test_local_on_disk_connection(tmp_path):
    # A filesystem path gives a persisted local client.
    client = get_client(str(tmp_path / "qdrant"))
    assert client.get_collections() is not None


@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="no QDRANT_URL server set")
def test_server_connection():
    # Runs only when a Qdrant server is reachable (QDRANT_URL set).
    client = get_client(os.environ["QDRANT_URL"])
    assert client.get_collections() is not None

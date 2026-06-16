"""Parser subpackages.

Each parser is a self-contained module that exposes:
    - ``parse(pdf_path: str) -> <native document>``
    - ``CHUNKERS: dict[str, Callable]`` — the chunkers compatible with *this*
      parser's native output.

There is intentionally no shared intermediate representation between parsers:
each chunker consumes its parser's native object directly. The only shared
contract is the ``Chunk`` output (see ``ingestion.schema``).
"""

"""Shared pytest setup: load .env so tests see the same environment as the CLI."""

from dotenv import load_dotenv

load_dotenv()

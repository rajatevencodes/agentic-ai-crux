# Agentic AI Crux Development Guide

## Stack

- Use Python 3.13, UV, and the official `openai` Python SDK.
- Call OpenRouter through its OpenAI-compatible API by configuring an `OpenAI` client with `base_url="https://openrouter.ai/api/v1"` and `OPENROUTER_API_KEY`.
- Do not add orchestration frameworks or provider-specific SDKs unless the user explicitly requests them.
- Use official OpenAI and OpenRouter documentation as the source of truth for their APIs.

## Notebook-first workflow rule

- Keep implementation and testing notebook-only unless the user explicitly requests source modules.
- Import reusable functions, environment loading, warning suppression, and logging configuration from `notebooks/common.py`.
- Create each `OpenAI` client explicitly in the notebook so its API key source, base URL, model ID, and request settings remain visible.
- After running the common setup cell, put every workflow step in its own cell with explicit sample input so it can be tested and rerun independently.
- Keep cells small; do not hide several state transitions behind one notebook call.
- Suppress warnings and set third-party loggers to `WARNING` or `ERROR` so output remains focused.

## Development

- Add dependencies with `uv add`, pin them with `uv add "package==version"`, and synchronize the environment with `make sync`.
- Start with ChromaDB when a learning exercise requires a vector database. Keep its persistence local and ignored by version control.
- Start with SQLite when a learning exercise requires local persistence. Keep the database local and ignored by version control.

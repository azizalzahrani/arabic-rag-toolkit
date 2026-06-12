# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-06-13

### Fixed
- **Embeddings**: constructor no longer crashes when `sentence-transformers` is installed but the model cannot be downloaded (offline, missing files). It now warns and falls back to the local hashing backend.
- **Embeddings**: `most_similar()` returned misaligned text/score pairs when the candidate list contained empty strings.
- **Embeddings**: `similarity()` could return NaN for zero-norm vectors; it now returns 0.0.
- **Generator**: requesting `llm_provider="anthropic"` no longer sends the OpenAI default model name to the Anthropic API. Each provider now has its own default (`gpt-4o-mini` / `claude-sonnet-4-6`), overridable via `model_name` or the `OPENAI_MODEL` / `ANTHROPIC_MODEL` environment variables.
- **Generator**: the Anthropic provider now honors `temperature` and `top_p`.
- **Generator / Retriever**: silent fallbacks to the local provider or in-memory store now emit a `RuntimeWarning` explaining why.
- **Retriever (Chroma)**: `load()` now reopens the persisted collection and restores documents, so search works after loading. Documents are also added in a single batch instead of one call per document.
- **Chunker**: the sentence/clause split pattern was built with `'|'.join(...)` inside a character class, making a literal `|` act as a sentence delimiter.
- **Chunker**: chunk overlap no longer cuts Arabic words in half; the overlap window now snaps to word boundaries.
- **Validator agent**: a similarity score of exactly 0.0 was treated as missing in the validation report average.
- **Writer agent**: `quality_score` could go negative with many issues; it is now clamped at 0.0.
- Corrected "معقم النصوص" to "مُقطِّع النصوص" across docstrings and docs.

### Changed
- **Original orthography is preserved in results**: documents are chunked as written and only the embedded copy is normalized. Retrieved snippets and answers now display the original hamzas and diacritics (e.g. "مسؤول" instead of "مسءول") while matching still happens on normalized text. Re-index existing stores after upgrading.
- Removed the nonstandard `ؤ -> ء` mapping from yaa normalization (alef maksura `ى -> ي` is kept).
- Normalization now also strips the dagger alef, maddah, and combining hamza marks, and folds alef wasla (`ٱ`) into plain alef.
- `pipeline.query(..., return_sources=True)` source lines now include a snippet of each document, not just the similarity score.
- `pipeline.reset()` keeps the loaded embedding model instead of reloading it.
- `GenerationConfig.model_name` now defaults to `None` (resolved per provider) instead of `"gpt-4"`.

### Added
- `ArabicRAGPipeline.from_env()` and `arabic_rag.pipeline.config_from_env()`: the variables documented in `.env.example` are now actually consumed by the package.
- `ArabicTextChunker(chunk_size=..., chunk_overlap=...)` keyword shortcuts, matching the documented examples.
- `ArabicRetriever.add_documents(documents, embedding_texts=...)` for separate display/embedding texts.
- `py.typed` marker so type checkers can use the package annotations.
- Python 3.13 in the CI matrix and PyPI classifiers; pip caching in CI.
- New test modules: `test_embeddings.py`, `test_generator.py`, plus regression tests for overlap boundaries, env config, and orthography preservation (30 -> 60 tests).

## [0.1.0] - 2026-03-22

- Initial release.

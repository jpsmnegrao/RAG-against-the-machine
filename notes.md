Stage 1 — Schemas (you're here). Get the pydantic models right and settled first, since every later stage passes data through them. Small, mechanical, low-risk — good first step.

Stage 2 — Chunking, in isolation. Write the two chunking strategies (Python-aware, Markdown-aware) as standalone functions you can unit-test on a handful of files by hand, before wiring them into any index or CLI. At this point, just eyeball: does a chunk boundary land somewhere sane? Are chunks under 2000 chars? This is worth getting right early since bad chunking caps your recall ceiling no matter how good retrieval scoring is later.

Stage 3 — Indexing. Wire chunking into something that walks data/raw/, produces chunks with correct file_paths (exact string match matters — get the path convention right immediately), and persists an index structure to data/processed/. At this point you can just print/inspect the index — no CLI yet necessarily, no scoring yet.

Stage 4 — Retrieval/scoring (BM25 or TF-IDF). Implement the scoring function that, given a query string and your index, returns ranked chunks. Test this directly against a few questions from the dataset you already have (you know the ground truth for those — use it as your own sanity check before even writing evaluate).

Stage 5 — Your own evaluate + recall@k. Now that you have real predictions and real ground truth, implement recall@k yourself (per the IoU/file_path rule) so you get a number, not just a vibe. This turns "search feels reasonable" into "recall@5 is 62%," which is what lets you actually iterate (chunk size, tokenization choices, BM25 vs TF-IDF, stopword handling, etc.) against a target.

Stage 6 — CLI wiring (Fire). Only once the underlying functions work when called directly in Python, wrap them behind Fire commands (index, search, search_dataset, evaluate). At this point it's just plumbing — argument parsing, path handling, save/load — not new logic.

Stage 7 — Hit the retrieval thresholds. Iterate specifically until you clear 80%/5 docs, 50%/5 code, within the time budgets. Don't move on until this is solid — this is the part with a hard numeric bar.

Stage 8 — Generation (Qwen). Only now bring the LLM in: prompt construction from retrieved chunks, calling Qwen3-0.6B, parsing/structuring its output into AnsweredQuestion/MinimalAnswer. Lower stakes, more forgiving grading, so it's fine to spend comparatively less iteration time here.

Stage 9 — Robustness pass. Deliberately throw edge cases at your CLI (empty query, k=0, missing files, malformed JSON, nonsense queries) and make sure nothing throws an unhandled traceback.

Stage 10 — README, Makefile polish, lint/mypy cleanup, final layout check. Do this last, once behavior is stable, so you're documenting/cleaning what actually exists rather than what you planned.
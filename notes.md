# Defense Prep — RAG against the machine

A study guide to review before the evaluation. Organized around the eval
sheet's actual checkpoints and the five "Student Understanding" questions,
with the reasoning behind each design choice spelled out so nothing needs to
be recalled from memory alone.

---

## 1. The five mandatory understanding questions

### "What is Retrieval-Augmented Generation (RAG) and why is it useful?"

A model only knows what it was trained on. Retraining it every time you need
new knowledge is slow and expensive. RAG sidesteps that: instead of putting
knowledge *into* the model's weights, you give the model access to an
external source at answer time. At query time you retrieve the most relevant
pieces of that source and hand them to the model as context, so it can
answer using information it was never trained on — a private codebase, a
document released yesterday, anything.

Four stages: **indexing** (organize the data so it's searchable), **retrieval**
(match a question against the index, pull relevant chunks), **augmenting**
(filter/place those chunks into the model's context window), **generation**
(the model reads that context and answers).

### "Walk me through your complete RAG pipeline, from raw documents to a generated answer."

1. **Indexing** (`index` command): `chunk_collector` walks `data/raw/`,
   finds every `.py`/`.md`/`.txt` file, and calls `chunkinizer` on each —
   which dispatches by extension to `file_chunker`, using
   `RecursiveCharacterTextSplitter.from_language` (Python-aware or
   Markdown-aware) or a generic splitter for `.txt`. Every chunk becomes an
   `IndexedChunk` (a `MinimalSource` — file path + character range — plus
   the chunk's own text). `bm25_scorer` tokenizes every chunk's text in one
   call (so the whole corpus shares one vocabulary) and builds a `bm25s.BM25`
   index. `index_exporter` pickles the index and the chunk list together
   under `data/processed/`.
2. **Retrieval** (`search`/`search_dataset`): loads the pickled index and
   chunks back into memory, tokenizes the query the same way, calls
   `BM25.retrieve()` to get the top-k ranked chunk positions, and maps those
   back to `MinimalSource` metadata.
3. **Generation** (`answer`/`answer_dataset`): re-reads the exact character
   span for each retrieved source directly from the original file (using the
   stored offsets), builds a prompt instructing the model to answer only
   from that context and cite the source file, and calls `Qwen/Qwen3-0.6B`
   to generate the answer.
4. **Evaluation** (`evaluate`): separately, compares a `search_dataset`
   output against a ground-truth dataset, computing recall@1/3/5/10.

### "What is the difference between TF-IDF and BM25 as retrieval methods?"

Both score how relevant a chunk is to a query using term frequency (TF) and
inverse document frequency (IDF) — TF rewards terms that appear often in a
chunk, IDF downweights terms that appear in almost every chunk (uninformative
words like "the") and upweights rare, distinctive terms.

BM25 refines plain TF-IDF in two ways:
- **Term-frequency saturation** — in TF-IDF, a term appearing 20 times in a
  chunk contributes roughly 20x the score of appearing once. BM25 applies
  diminishing returns instead, so extra repetitions matter less and less —
  closer to how relevance actually behaves (a chunk repeating a word 20
  times isn't 20x more relevant than one that mentions it once).
- **Document-length normalization** — TF-IDF doesn't correct for chunk
  length, so a longer chunk can score artificially high just from having
  more words overall. BM25 explicitly normalizes against the corpus's
  average chunk length, so long chunks aren't unfairly favored.

I implemented BM25 (via the `bm25s` library) rather than TF-IDF because it's
generally the stronger, more standard baseline in modern lexical search
(it's what a lot of real search engines use as a default), and the subject
only requires implementing at least one of the two.

### "What implementation choices did you make and why? What trade-offs did you consider?"

See section 4 below (Design Decisions) — walk through: library-based
chunking over hand-rolled AST splitting, single combined pickle for the
index, `evaluate` reimplementing the moulinette's exact logic, graceful
degradation over hard failures, and the file-path-injection experiment that
was tried but reverted.

### "If you had more time, what would you improve in your system?"

- Pursue the semantic-embeddings / hybrid-retrieval bonus, since BM25 is
  purely lexical and structurally can't bridge natural-language questions to
  code identifiers that don't share vocabulary — this is exactly why code
  recall (55.6%) trails docs recall (83.0%).
- The file-path-injection experiment (prepending each chunk's file path to
  the text before tokenizing) measurably improved recall on both datasets
  when tested (docs Recall@5 83%→86%, code Recall@5 55.6%→58.6%) but was
  reverted for time/focus reasons — worth re-adopting with more time to
  validate it doesn't have any downside I didn't test for.
- Speed up answer generation — the campus GPU's small VRAM (~3.6 GB) meant
  generation was slow even after disabling Qwen3's reasoning trace; running
  inference through vLLM itself (one of the project's own bonuses) is the
  most promising untried lever, though a peer report suggested it may not
  even install cleanly on the same campus hardware.
- Expand automated test coverage of CLI-level edge cases — several real bugs
  (see section 6) were only found by manually running the CLI with garbage
  input, after the existing automated tests had already passed; a more
  adversarially-designed test suite, written before rather than around the
  code, would have caught these mechanically instead.

---

## 2. Chunking strategy — what the corrector will ask

**"What happens if chunks are too small? Too large?"**

- **Too large**: less precise retrieval — a big chunk might get retrieved
  because *one* sentence in it matches the query, but the rest of the chunk
  is irrelevant noise handed to the model as context, and the model's
  limited context window fills up faster with less useful content per
  token. The subject also hard-caps this: any retrieved source over 2000
  characters is rejected by the moulinette outright.
- **Too small**: chunks may lose surrounding context — a chunk containing
  just one line of a function's body, with no signature or context, is
  syntactically meaningless in isolation and less likely to be a genuinely
  useful retrieval unit; you also get many more chunks, each carrying less
  independent value, and the index grows without necessarily improving
  recall.

**The two implemented strategies:**
- **Python (`.py`)**: `RecursiveCharacterTextSplitter.from_language` with
  `Language.PYTHON` — prefers splitting at class/function boundaries first,
  falling back to blank lines, then single lines, then raw characters if a
  block still exceeds `max_chunk_size`.
- **Markdown (`.md`)**: the same splitter with `Language.MARKDOWN` — prefers
  headers and paragraph breaks.
- **Plain text (`.txt`)**: a generic splitter with no language-specific
  separators, since the subject's own wording groups "Markdown / text
  chunking" as one strategy.

All three are capped at `max_chunk_size` (default 2000, configurable via
`--max_chunk_size`), with `chunk_overlap = max_chunk_size // 10` so content
straddling a split boundary isn't entirely lost from both resulting chunks.

**Why a library instead of hand-writing an AST-based splitter:** an
AST-based chunker (using Python's `ast` module) was actually prototyped
first — computing character offsets from `ast.FunctionDef`/`ClassDef` node
line numbers, splitting on top-level definitions. This was valuable for
understanding the underlying mechanics (how to convert line numbers to
character offsets, how to decide what counts as one "chunk"), but once that
was understood, chunking was rebuilt on `RecursiveCharacterTextSplitter`
because: the grading metric's IoU threshold (0.05) doesn't require precise
chunk boundaries, so the extra precision of a hand-built AST splitter isn't
rewarded by the metric; and the library already implements the fallback
logic (oversized-block splitting) that would otherwise have to be built by
hand.

---

## 3. Retrieval — what the corrector will ask

**"Explain how your chosen method works"** — see the BM25/TF-IDF answer
above.

**Live demo command**: `uv run python -m src search "How to configure OpenAI
server?" --k 10` — be ready to explain each printed line: it's a
`file_path` plus a `[start:end]` character range, the exact
`MinimalSource` shape the grader compares.

**How results are produced, end to end:** the query is tokenized the same
way the corpus was tokenized at index time (same `bm25s.tokenize` call, no
special-casing), `BM25.retrieve()` scores it against every indexed chunk and
returns the top-k positions, and those positions are mapped back to the
`corpus` list saved alongside the index — this is why chunks and their
tokenized text must always stay in the same order and never be reordered
independently after being built once.

---

## 4. Design decisions and trade-offs

1. **Library-based chunking over hand-rolled AST splitting** — see section 2.
2. **A single combined pickle file pair for the index** (`index.pkl` +
   `corpus.pkl`) rather than using `bm25s`'s own `save(..., corpus=...)`
   convenience or separate ad-hoc formats — chosen because it avoids ever
   loading a mismatched pair (chunks built from a different run than the
   index they're paired with), and pickle preserves the actual pydantic
   objects natively without needing a dict-conversion round trip.
3. **`evaluate` re-implements the moulinette's own logic exactly**, not a
   simplified approximation — same file-path exact-match rule, same IoU ≥
   0.05 overlap rule, same recall@1/3/5/10 reporting — verified to produce
   numbers identical to the official moulinette on both public datasets, so
   it's trustworthy for iteration before running the real grading tool.
4. **Graceful degradation over hard failures everywhere in the CLI** —
   missing index files, missing source files, malformed JSON, non-numeric or
   boolean CLI flags, empty/whitespace queries, and negative/zero `k` are all
   validated and handled with a clear printed message rather than an
   unhandled traceback, matching the subject's explicit requirement that the
   CLI "must never crash with an unhandled traceback."
5. **BM25 over TF-IDF** — see section 1's answer above.
6. **File-path injection into indexed text — tried, measured, reverted for
   now.** Prepending each chunk's file path to its text before tokenizing
   (so BM25's lexical matching can hit on filenames/identifiers mentioned in
   a question) measurably improved recall on both datasets when tested.
   Left out of the current submission to focus remaining time elsewhere, but
   the before/after numbers are known and it's a clear, low-risk candidate
   for a follow-up improvement.

---

## 5. Challenges faced — concrete stories worth telling

**A single-character metric bug, found by disciplined debugging, not luck.**
An early version of `evaluate` used an IoU threshold of `0.5` instead of the
spec's `0.05`. This silently under-counted every partial-overlap match,
producing recall numbers noticeably lower than the moulinette's own — but
without crashing or throwing any error, since the code was otherwise
correct. It was found by: comparing `evaluate`'s output against the
moulinette's output on the same input files; picking a question where the
two disagreed; manually computing the ground-truth IoU by hand for that
question's actual character ranges; confirming that hand-computed value was
above the intended threshold; and only then realizing the *threshold in the
code*, not the IoU formula itself, was wrong. Worth mentioning as a genuine
example of methodical debugging rather than guessing.

**Small-GPU hardware constraints on the campus machine.** The available GPU
has very little VRAM (~3.6 GB). Loading Qwen3-0.6B at full or half precision
caused out-of-memory errors; 4-bit quantization helped fit the model but
generation was still slow; CPU-only generation was also slow. The single
largest, reliable speed improvement came from disabling Qwen3's default
`<think>` reasoning trace (`enable_thinking=False`), which was consuming the
bulk of generation time on a reasoning pass the answer didn't need. This was
treated as a hardware ceiling, not a code defect, since the subject sets no
time budget for answer generation specifically (only for indexing and
retrieval).

**A test suite that passed while real bugs remained.** Several CLI
robustness bugs — a `TypeError` crash when `--k` received a non-numeric
string (caused by a type check running *after* a numeric comparison instead
of before it), a `raise` in `search_dataset` that crashed on a missing
`--save_directory` value instead of failing gracefully like every other
guard in the file, and multiple parameters silently coercing to `True`/`1`
when a flag was given with no value — were only found by manually running
the CLI with deliberately malformed input, *after* the existing automated
test suite had already passed in full. The tests that existed were correct
for the specific values they checked (`k=0`, `k=-1`, an empty string) but
never covered the *category* of "wrong type entirely" or "flag with no
value." Worth being honest about this if asked what you'd improve: a more
adversarial test-writing process, designed before the code rather than
around it, would have caught these mechanically.

**Keeping character offsets consistent through library-based chunking.**
Since chunking is delegated to `RecursiveCharacterTextSplitter`, care was
taken to rely only on its own `start_index` metadata (via
`add_start_index=True`) rather than re-deriving offsets independently, and
to verify with unit tests that `raw_text[start:end]` always equals the
chunk's own stored text exactly — since any drift here would silently
corrupt every downstream retrieval and grading comparison.

---

## 6. Recall@k and IoU — explained simply, with a worked example

**Recall@k** answers: "out of the correct source(s) for this question, how
many did the top-k retrieved results actually include?" For one question,
recall = (correct sources found in top-k) / (total correct sources for that
question). Averaged across all questions in a dataset, this gives one
overall recall@k score.

**A retrieved source only counts as "found" if two conditions both hold:**
1. `file_path` matches the ground truth **exactly**, character for
   character (a different file, or a different path prefix, never counts —
   no partial credit).
2. The character range **overlaps** the ground truth's character range with
   an IoU of at least 0.05.

**IoU (Intersection over Union), worked example**, using a real case found
during debugging: ground truth range `[3619, 4765]`, retrieved range
`[3820, 5739]`, same file.

- Intersection = the overlapping portion = `min(4765, 5739) - max(3619,
  3820) = 4765 - 3820 = 945`.
- Union = the combined span both ranges cover = `max(4765, 5739) -
  min(3619, 3820) = 5739 - 3619 = 2120`.
- IoU = `945 / 2120 ≈ 0.445`.

Since `0.445 ≥ 0.05`, this counts as a hit — the threshold is deliberately
low, so a chunk doesn't need to align tightly with the reference span, just
land somewhere genuinely overlapping the right region of the right file.

---

## 7. Pydantic models — the seven required, plus one extension

- `MinimalSource` — `file_path`, `first_character_index`,
  `last_character_index`. One located span of text.
- `UnansweredQuestion` — `question_id` (auto-generated UUID if not given),
  `question`. What a test-time question looks like, no answer key attached.
- `AnsweredQuestion(UnansweredQuestion)` — adds `sources`, `answer`. The
  ground-truth format.
- `RagDataset` — `rag_questions: list[AnsweredQuestion | UnansweredQuestion]`.
  The container for a whole dataset file.
- `MinimalSearchResults` — `question_id`, `question`, `retrieved_sources`.
  One question's worth of retrieval predictions.
- `MinimalAnswer(MinimalSearchResults)` — adds `answer`. Retrieval plus a
  generated answer.
- `StudentSearchResults` — `search_results: list[MinimalSearchResults]`,
  `k`. The file written by `search_dataset`.
- `StudentSearchResultsAndAnswer` — `search_results: list[MinimalAnswer]`,
  `k`. The file written by `answer_dataset`.
- **`IndexedChunk`** (project extension, not in the original spec) —
  `minimalsource: MinimalSource`, `text: str`. Needed because
  `MinimalSource` alone carries only file path and offsets, not the actual
  chunk text the indexer needs to tokenize; the subject explicitly allows
  extending the provided models where the implementation requires it.

---

## 8. Tooling choices — quick justifications if asked

- **`uv`** — required by the subject; the moulinette and reviewer only run
  `uv sync`.
- **Python Fire** — chosen because it turns plain functions into CLI
  commands automatically from their signatures, with no manual argument
  parser to write per command, unlike `argparse`. Required by the subject
  regardless.
- **`bm25s`** — a modern, numpy/scipy-backed BM25 implementation, faster
  than older pure-Python alternatives (like `rank_bm25`), which matters
  given the 90-second/200-question retrieval budget.
- **`pydantic`** — required by the subject for all data interchanged
  between stages; also gives free validation (e.g. `evaluate`'s own
  "Student data is valid" check is a real schema validation, not just a
  non-empty check).
- **`tqdm`** — required by the subject for long-running operations;
  wrapped around the corpus-walk/chunking loop, the tokenization step, and
  the batch search/answer loops.

---

## 9. Things to physically demo if asked

- `uv run python -m src index --max_chunk_size 2000` — show the two
  progress bars and the final "Ingestion complete!" line.
- `uv run python -m src search "How to configure OpenAI server?" --k 10` —
  show ranked results, explain the offsets.
- `uv run python -m src answer "How to configure OpenAI server?" --k 10` —
  show a full grounded answer with source citation.
- `uv run python -m src evaluate --student_search_results_path ... --dataset_path ...`
  — show it reproduces the moulinette's own numbers.
- Edge cases, if asked: empty query, `k=0`, a nonexistent dataset path — all
  should print a clear message and exit cleanly, no traceback.



git switch -c feature/my-feature

git add .
git commit -m ""
git push -u origin branchname

git switch main
git pull
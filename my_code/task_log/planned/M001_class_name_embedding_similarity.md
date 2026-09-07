# M001: Embed class names and export cosine similarities

## Metadata

- ID: M001
- Title: Embed class names and export cosine similarities
- Type: Code implementation task
- Status: IN_PROGRESS
- Implementation: `my_code/data/OpenImage/meta/class_name_similarity.py`
- Input: `my_code/data/OpenImage/meta/Class Names Map.csv`
- Input columns: `LabelName`, `DisplayName`
- Observed input size: 20,931 data rows plus header
- Embedding model: `BAAI/bge-m3`

## Scope

Allowed implementation files:

- `my_code/data/OpenImage/meta/class_name_similarity.py`
- `requirements-training.txt` or a more appropriate existing dependency tier
- `my_code/data/OpenImage/meta/README_class_name_similarity.md`

Generated artifacts must be written outside version control or to an explicitly
requested output directory. Do not commit embeddings, matrices, caches,
checkpoints, or downloaded model files.

Forbidden changes:

- No changes to `src/open_clip/`, `src/training/`, or unrelated `my_code`
  utilities.
- No modification of the source CSV.
- No hardcoded access tokens.

## Required sub-tasks

1. Define a CLI with input path, output directory, model name, batch size,
   device, and optional max sequence length.
2. Load `DisplayName` with UTF-8 CSV handling, preserve source row order and
   `LabelName`, and validate required columns and non-empty class names.
3. Load `BAAI/bge-m3` through a maintained embedding interface, support CPU
   and CUDA selection, normalize embeddings before similarity calculation, and
   batch inference to avoid unnecessary memory spikes.
4. Compute the full pairwise cosine similarity matrix, keep diagonal values
   out of nearest-neighbor selection, and select the top three distinct other
   class names in descending similarity order with deterministic tie handling.
5. Export a nearest-neighbor table containing the original identifiers and
   class name plus `Top1Name`, `Top1Cosine`, `Top2Name`, `Top2Cosine`,
   `Top3Name`, and `Top3Cosine`.
6. Export the complete square cosine matrix with stable row and column class
   ordering, preferably with identifiers that remain unambiguous if display
   names repeat.
7. Add resumability or a reusable embedding cache when practical, while
   keeping cache and output paths configurable and excluded from git.
8. Document runtime requirements, model download behavior, expected output
   schemas, and an estimate of the matrix storage cost for this input size.

## Acceptance criteria

- Running the CLI on the supplied CSV produces both requested output tables.
- Every input row appears exactly once in the nearest-neighbor table in source
  order, and each row has three other classes unless fewer than four distinct
  valid classes exist.
- The similarity matrix is square, symmetric within floating-point tolerance,
  has ones on the diagonal within tolerance, and uses the same class ordering
  in both dimensions.
- Reported top-three scores equal the corresponding matrix cells within the
  documented numeric precision.
- Duplicate display names are handled deterministically and do not cause
  self-selection by row identity.
- No model weights, generated CSVs, or embedding caches are added to git.
- The implementation passes `make build` and `make lint-arch`; a focused smoke
  run verifies CSV parsing and ranking with a tiny synthetic fixture, while a
  full model run is documented separately if model download or hardware is not
  available.

## Validation commands

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python my_code/data/OpenImage/meta/class_name_similarity.py --help
make build
make lint-arch
```

Use a temporary four-to-six-row fixture for the offline ranking smoke test.
Run the full 20,931-class job only in an environment with `sentence-transformers`
or the selected embedding dependency, model download access, and sufficient
storage for the dense matrix. The implementation must report the exact output
paths and row/column counts on completion.

## Output contract

Recommended filenames:

- `class_name_nearest_neighbors.csv`
- `class_name_cosine_similarity_matrix.csv`

The matrix must state its ordering convention. If CSV size or duplicate names
make a flat matrix unwieldy, retain the requested matrix as a documented binary
array artifact in addition to the human-readable CSV rather than silently
omitting it.

## Notes and risks

- A 20,931 by 20,931 float32 matrix requires about 1.63 GiB before CSV text
  overhead, so CSV export can be substantially larger and slower.
- The bge-m3 model is external to the current TinyCLIP dependency set and may
  require a new dependency plus a one-time download.
- The input contains human-readable names with punctuation; CSV quoting and
  Unicode round-tripping must be tested.


## Implementation record

- Added a CLI that loads the source CSV, embeds names with BAAI/bge-m3, caches embeddings, normalizes vectors, computes cosine similarities, ranks Top-3 neighbors, and exports CSV/NumPy outputs.
- Added dependency ``sentence-transformers>=2.7.0`` to ``requirements-training.txt``.
- Offline ranking and matrix smoke test passed with synthetic vectors.
- Full model execution remains environment-dependent because sentence-transformers and model weights are not installed in the current environment.



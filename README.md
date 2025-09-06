# codesimilarity

Compare similarity of Python code.

## Usage

```bash
uvx --from "git+https://github.com/sanand0/codesimilarity.git" codesimilarity a.py b.py c/
```

This computes pairwise Jaccard overlap of k-token phrases between `a.py`, `b.py` and all `.py` files under `c/` (treated as a single concatenated document).

## How It Works

- Tokenizes Python via `tokenize`, ignoring whitespace, comments, and indentation.
- Builds shingles: all `k`-length sequences of tokens per input.
- Computes Jaccard similarity: `|A ∩ B| / |A ∪ B|` on shingle sets.

## CLI

```bash
uvx --from "git+https://github.com/sanand0/codesimilarity.git" \
  codesimilarity PATH [PATH ...] \
  --csv out.csv \
  --threshold 0.0 \
  --k 5 \
  --nearest 0 \
  --lexical
```

PATH handling:

- Python files (`.py`) are processed.
- Directories are processed as a single document made by concatenating all `.py` files under them (recursively).
- Non-Python files are ignored with a warning: `Warning: ignoring non-Python file: <path>`.
- Missing paths warn: `Warning: path not found: <path>`.

Options:

- `--k` sets phrase length. Larger `k` reduces incidental matches but misses re-orderings. 5-8 is good.
- `--lexical` compare exact literals. Treats string / number changes as different.
- `--threshold` filters out pairs below the given overlap (applies only in pairwise mode).
- `--nearest N` outputs only the top-N nearest matches per input (disables pairwise mode).
- `--csv out.csv` writes CSV output to the given file; otherwise prints TSV to stdout.

Output modes:

- Pairwise (default): rows for all ordered pairs, with columns `left,right,overlap`.
  - `--csv out.csv` writes CSV with 6-decimal floats; otherwise prints TSV to stdout with 3 decimals.
- Nearest summary (`--nearest N`): one row per input showing the nearest matches. Columns are:
  - max_overlap
  - mean_overlap
  - nearest_1,... nearest_N
  - overlap_1,... overlap_N.

## Literal Normalization (`--lexical`)

By default, replaces all strings with `STRING` and all numbers with `NUMBER`. Identifies (variable/function/class names) are not normalized.

So, even if someone changes a string or number value, it won't affect the similarity score.

Example:

```python
# a.py
def f(a, b):
    s = "hello"
    return a + b

# b.py
def f(a, b):
    s = "world"
    return a + b
```

- With default normalization, the overlap is 1.0.
- With `--lexical`, the overlap is 0.375 because the string literals differ.

## Phrase Size (`--k`)

Larger `k` reduces incidental matches. For short snippets, very large `k` can lead to zero overlap.

Example:

```python
# mul.py
def mul(a, b=2):
    return a * b
```

With `--k 2`, we find some incidental overlap between `add1.py` and `mul.py` because they share short phrases like `def <NAME>` and `( <NAME>`.

```bash
uvx --from "git+https://github.com/sanand0/codesimilarity.git" codesimilarity *.py --k 2
```

| left    | right   | jaccard |
| ------- | ------- | ------: |
| add1.py | add2.py |   0.714 |
| add1.py | mul.py  |   0.333 |
| add2.py | add1.py |   0.714 |
| add2.py | mul.py  |   0.368 |
| mul.py  | add2.py |   0.368 |
| mul.py  | add1.py |   0.333 |

With `--k 5`, the incidental overlap disappears, and `add1.py` and `mul.py` have no 5-token phrases in common.

```bash
uvx --from "git+https://github.com/sanand0/codesimilarity.git" codesimilarity *.py --k 5
```

| left    | right   | jaccard |
| ------- | ------- | ------: |
| add1.py | add2.py |   0.286 |
| add1.py | mul.py  |   0.000 |
| add2.py | mul.py  |   0.286 |
| add2.py | add1.py |   0.053 |
| mul.py  | add2.py |   0.053 |
| mul.py  | add1.py |   0.000 |

## Development

This repository is not published on PyPI. Develop and test locally:

- Clone: `git clone https://github.com/sanand0/codesimilarity && cd codesimilarity`
- Create env: `uv venv && source .venv/bin/activate`
- Install deps: `uv pip install -e .[dev]`
- Run tests: `uv run pytest`

Notes:

- The CLI is exposed via the `codesimilarity` entry point (Typer). Use `uvx codesimilarity ...` as shown above.

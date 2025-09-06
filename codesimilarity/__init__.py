"""codesimilarity: Compare similarity of Python code."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Set
from io import BytesIO

import pandas as pd
import tokenize
import token as tok
import typer


DROP_TYPES = {
    # layout/EOF only; no semantic signal for plagiarism
    tok.ENCODING,
    tok.NL,
    tok.NEWLINE,
    tok.INDENT,
    tok.DEDENT,
    tok.ENDMARKER,
    # easily edited; high-noise, low-signal
    tok.COMMENT,
    tok.TYPE_COMMENT,
    tok.TYPE_IGNORE,
    # rare, parser rejects
    tok.ERRORTOKEN,
    # style/rarity; provide little discriminative value vs structure
    tok.ELLIPSIS,
    tok.SEMI,
}
NORMALIZE = {
    tok.STRING: "STRING",
    tok.NUMBER: "NUMBER",
}


def tokenize_code(code: str, normalize: bool = False) -> List[str]:
    """Tokenize Python code into words, ignoring whitespaces, comments, etc."""
    result: List[str] = []
    reader = BytesIO(code.encode("utf-8")).readline
    for t in tokenize.tokenize(reader):
        if t.type in DROP_TYPES:
            continue
        elif normalize and t.type in NORMALIZE:
            result.append(NORMALIZE[t.type])
        else:
            result.append(t.string)
    return result


def shingles(tokens: Sequence[str], k: int) -> Set[Tuple[str, ...]]:
    """Return set of k-length token tuples."""
    return {tuple(tokens[i : i + k]) for i in range(0, max(0, len(tokens) - k + 1))}


def overlap(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    """Compute Jaccard similarity between two shingle sets."""
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _read_py(arg: str, k: int, normalize: bool) -> Tuple | None:
    path = Path(arg)
    if path.is_dir():
        contents: List[str] = []
        for path in sorted(path.rglob("*.py")):
            contents.append(path.read_text(encoding="utf-8", errors="ignore"))
        tokens = tokenize_code("\n".join(contents), normalize)
        return (arg, shingles(tokens, k)) if len(tokens) > 0 else None
    if path.is_file() and path.suffix == ".py":
        content = path.read_text(encoding="utf-8", errors="ignore")
        tokens = tokenize_code(content, normalize)
        return (arg, shingles(tokens, k)) if len(tokens) > 0 else None
    if path.exists():
        typer.echo(f"Warning: ignoring non-Python file: {arg}", err=True)
    else:
        typer.echo(f"Warning: path not found: {arg}", err=True)
    return None


def _pairwise_similarity(docs: List[Tuple]) -> pd.DataFrame:
    """Return full pairwise Jaccard as a DataFrame (index/columns = doc names)."""
    pairs = []
    for left_name, left_shingles in docs:
        for right_name, right_shingles in docs:
            if left_name != right_name:
                pairs.append((left_name, right_name, overlap(left_shingles, right_shingles)))
    return pd.DataFrame(pairs, columns=["left", "right", "overlap"])


def nearest_matches(group, nearest):
    top = group.nlargest(nearest, "overlap")
    result = {
        "max_overlap": group["overlap"].max(),
        "mean_overlap": top["overlap"].mean(),
        **{f"nearest_{i + 1}": b for i, b in enumerate(top["right"])},
        **{f"overlap_{i + 1}": j for i, j in enumerate(top["overlap"])},
    }
    return pd.Series(result)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    paths: List[str] = typer.Argument(..., help="Files or directories to compare"),
    lexical: bool = typer.Option(False, "--lexical", help="Don't normalize literals"),
    threshold: float = typer.Option(0.0, "--threshold", help="Ignore pairs below this overlap"),
    k: int = typer.Option(5, "--k", help="Shingle size (phrase length)"),
    nearest: int = typer.Option(0, "--nearest", help="Top-N most similar per input"),
    csv: Path | None = typer.Option(None, "--csv", help="Save as CSV"),
) -> None:
    """Compute pairwise code similarity and write CSV outputs."""

    docs = [doc for doc in (_read_py(path, k, not lexical) for path in paths) if doc is not None]
    if not docs:
        typer.echo("No valid Python inputs found.", err=True)
        raise typer.Exit(code=2)

    pairs = _pairwise_similarity(docs)
    # Drop pairs below threshold. Sort by highest overlap first.
    result = pairs[pairs["overlap"] >= threshold].sort_values(["overlap"], ascending=False)

    # If nearest mode, compute nearest-N summary per input.
    if nearest:
        result = pairs.groupby("left").apply(nearest_matches, nearest, include_groups=False)
        result = result.sort_values("mean_overlap", ascending=False)

    # Save CSV or print TSV to stdout.
    if csv:
        result.to_csv(csv, index=False, float_format="%.6f")
    else:
        output = BytesIO()
        result.to_csv(output, index=False, float_format="%.3f", sep="\t")
        typer.echo(output.getvalue().decode("utf-8"))


if __name__ == "__main__":  # pragma: no cover - entry for `python codesimilarity/__init__.py`
    app()

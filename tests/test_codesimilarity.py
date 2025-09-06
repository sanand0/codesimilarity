from __future__ import annotations

from pathlib import Path
import csv
from typer.testing import CliRunner

from codesimilarity import app, tokenize_code


runner = CliRunner()


def write(p: Path, s: str) -> Path:
    p.write_text(s, encoding="utf-8")
    return p


def test_tokenize_ignores_comments() -> None:
    code = """
x = 1  # comment
s = "hello"
print("x", x)
"""
    tokens = tokenize_code(code)
    assert tokens == [
        "x",
        "=",
        "1",
        "s",
        "=",
        '"hello"',
        "print",
        "(",
        '"x"',
        ",",
        "x",
        ")",
    ]


def test_tokenize_normalizes_and_ignores_comments() -> None:
    code = """
x = 1  # comment
s = "hello"
print("x", x)
"""
    tokens = tokenize_code(code, normalize=True)
    assert tokens == [
        "x",
        "=",
        "NUMBER",
        "s",
        "=",
        "STRING",
        "print",
        "(",
        "STRING",
        ",",
        "x",
        ")",
    ]


def test_cli_pairwise_and_nearest(tmp_path: Path) -> None:
    a = write(tmp_path / "a.py", "def add(a,b):\n    return a + b\n")
    b = write(tmp_path / "b.py", "def add(a,b):\n    return a+b\n")
    cdir = tmp_path / "c"
    cdir.mkdir()
    write(cdir / "c1.py", "def sub(a,b):\n    return a - b\n")
    write(cdir / "c2.py", "x=1\nprint(x)\n")
    txt = write(tmp_path / "d.txt", "not python")
    # Pairwise CSV output
    pairs_csv = tmp_path / "pairs.csv"
    result = runner.invoke(
        app,
        [
            str(a),
            str(b),
            str(cdir),
            str(txt),  # should be ignored with a warning
            "--csv",
            str(pairs_csv),
            "--threshold",
            "0.0",
            "--k",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: ignoring non-Python file" in result.stderr

    with pairs_csv.open() as f:
        rows = list(csv.DictReader(f))
    # Expect all ordered pairs for 3 docs => 6 rows
    assert len(rows) == 6
    lefts = {r["left"] for r in rows}
    rights = {r["right"] for r in rows}
    assert lefts == {str(a), str(b), str(cdir)}
    assert rights == {str(a), str(b), str(cdir)}
    # Ensure a != b per row and overlap is a float string
    for r in rows:
        assert r["left"] != r["right"]
        assert 0.0 <= float(r["overlap"]) <= 1.0

    # Nearest summary (no 'left' column in current behavior)
    nearest_csv = tmp_path / "nearest.csv"
    result2 = runner.invoke(
        app,
        [
            str(a),
            str(b),
            str(cdir),
            "--csv",
            str(nearest_csv),
            "--nearest",
            "1",
        ],
    )
    assert result2.exit_code == 0, result2.output
    with nearest_csv.open() as f:
        nrows = list(csv.DictReader(f))
    # One row per input, summarised
    assert len(nrows) == 3
    for r in nrows:
        # left is index-only and not written; ensure expected summary columns exist
        assert set(r.keys()) >= {"max_overlap", "mean_overlap", "nearest_1", "overlap_1"}
        assert r["nearest_1"] in {str(a), str(b), str(cdir)}
        o1 = float(r["overlap_1"]) if r["overlap_1"] else 0.0
        mx = float(r["max_overlap"]) if r["max_overlap"] else 0.0
        mn = float(r["mean_overlap"]) if r["mean_overlap"] else 0.0
        assert 0.0 <= o1 <= 1.0
        # With nearest=1, mean == max == overlap_1
        assert abs(o1 - mx) < 1e-9
        assert abs(o1 - mn) < 1e-9


def test_threshold_and_k_effect(tmp_path: Path) -> None:
    x = write(tmp_path / "x.py", "x=1\n")
    y = write(tmp_path / "y.py", "y=2\n")
    pairs_csv = tmp_path / "pairs.csv"
    # k larger than token count -> 0 overlap off-diagonal; threshold filters out
    res = runner.invoke(
        app,
        [
            str(x),
            str(y),
            "--csv",
            str(pairs_csv),
            "--k",
            "5",
            "--threshold",
            "0.1",
        ],
    )
    assert res.exit_code == 0
    with pairs_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == []  # nothing >= 0.1


def test_samples_similarity_and_k_variation(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    a1 = write(
        samples / "add1.py",
        """
def add(a, b):
    s = "hello"
    return a + b
""",
    )
    a2 = write(
        samples / "add2.py",
        """
def add(a,b):
    s = "world"  # different string; normalized
    return a+b
""",
    )
    mul = write(
        samples / "mul.py",
        """
def mul(a,b):
    return a*b
""",
    )

    # k=2 should be more permissive than k=5
    def run_k(k: int, pairs_path: Path) -> dict[tuple[str, str], float]:
        res = runner.invoke(
            app,
            [
                str(a1),
                str(a2),
                str(mul),
                "--csv",
                str(pairs_path),
                "--k",
                str(k),
            ],
        )
        assert res.exit_code == 0, res.output
        out: dict[tuple[str, str], float] = {}
        for row in csv.DictReader(pairs_path.open()):
            out[(row["left"], row["right"])] = float(row["overlap"]) if row["overlap"] else 0.0
        return out

    pairs2 = run_k(2, tmp_path / "pairs_k2.csv")
    pairs5 = run_k(5, tmp_path / "pairs_k5.csv")

    key_a1_a2 = (str(a1), str(a2))
    key_a1_mul = (str(a1), str(mul))
    assert pairs2[key_a1_a2] > pairs2.get(key_a1_mul, 0.0)
    # With larger k, similarity should not increase
    assert pairs5[key_a1_a2] <= pairs2[key_a1_a2]


def test_prints_pairwise_to_stdout(tmp_path: Path) -> None:
    a = write(tmp_path / "a.py", "print('x')\n")
    b = write(tmp_path / "b.py", "print('y')\n")
    res = runner.invoke(app, [str(a), str(b)])
    assert res.exit_code == 0

    # Should print tab-delimited pairwise output to stdout
    lines = res.stdout.strip().split("\n")
    assert lines[0] == "left\tright\toverlap"
    assert lines[1].startswith(f"{str(a)}\t{str(b)}\t")
    assert lines[2].startswith(f"{str(b)}\t{str(a)}\t")


def test_cli_lexical_mode_affects_similarity(tmp_path: Path) -> None:
    """--lexical disables literal normalization, reducing similarity when literals differ."""
    a = write(
        tmp_path / "a.py",
        """
def f(a, b):
    s = "hello"
    return a + b
""",
    )
    b = write(
        tmp_path / "b.py",
        """
def f(a, b):
    s = "world"  # different literal
    return a + b
""",
    )

    def run_args(extra: list[str], out_path: Path) -> float:
        res = runner.invoke(
            app,
            [
                str(a),
                str(b),
                "--csv",
                str(out_path),
                "--k",
                "3",
                *extra,
            ],
        )
        assert res.exit_code == 0, res.output
        rows = list(csv.DictReader(out_path.open()))
        # Two docs -> two directional rows; take a->b
        row = next(r for r in rows if r["left"] == str(a) and r["right"] == str(b))
        return float(row["overlap"]) if row["overlap"] else 0.0

    j_norm = run_args([], tmp_path / "pairs_norm.csv")
    j_lex = run_args(["--lexical"], tmp_path / "pairs_lex.csv")

    # Normalized (default) should be at least as similar as lexical; typically 1.0 > x
    assert j_norm >= j_lex
    assert j_lex < 1.0

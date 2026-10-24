"""Merge the task notebooks into ONE Jupyter submission notebook.

Simple version: cells copied VERBATIM in fixed notebook order (00 -> 07).
The only changes:
- the per-notebook `import marimo as mo` shim: the first one is kept (it
  provides `mo` for every later markdown cell); the rest are dropped;
- an import statement already covered by an earlier import = skipped
  (or shrunk to its not-yet-imported names).
Load assignments stay IN their places — plain Jupyter allows rebinding;
the loads are cached parquet reads (seconds).

Usage:
    uv run python scripts/build_submission.py        # build
    uv run python scripts/build_submission.py --run  # + integration test
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOKS = ["00_setup", "01_count_words", "02_topics", "03_feelings",
             "04_personality_buckets", "05_meaning_map", "06_ai_explains",
             "07_authors_method"]
SRC_DIR = Path("notebooks")
OUT_DIR = Path("reports")


def export_ipynb(name, tmp):
    """One notebook -> parsed ipynb via marimo's own exporter."""
    out = tmp / f"{name}.ipynb"
    subprocess.run(
        ["uv", "run", "marimo", "export", "ipynb",
         str(SRC_DIR / f"{name}.py"), "-o", str(out)],
        check=True, capture_output=True,
    )
    return json.loads(out.read_text())["cells"]


def import_pairs(statement):
    """(module_root, local_name) pairs one import statement provides."""
    head = ast.parse(statement).body[0]
    if isinstance(head, ast.Import):
        return tuple(sorted({
            (a.name.split(".")[0], a.asname or a.name) for a in head.names
        }))
    module = "." * head.level + (head.module or "")
    return tuple(sorted({
        (module, a.asname or a.name) for a in head.names
    }))


def reemit(statement, new_pairs):
    """A from-import shrunk to its not-yet-covered names, keeping the
    alias form when the local name differs from the imported one."""
    head = ast.parse(statement).body[0]
    if isinstance(head, ast.ImportFrom):
        module = "." * head.level + (head.module or "")
        keep = {
            (a.asname or a.name): a.name for a in head.names
        }
        parts = [
            f"{original} as {alias}" if alias != original else original
            for alias, original in sorted(keep.items())
            if (module, alias) in new_pairs
        ]
        return f"from {module} import " + ", ".join(parts)
    return statement  # plain imports: all-or-nothing (handled by caller)


def bound_names(assignments_source):
    """Assignment/loop target names of one code cell (shadow warnings)."""
    tree = ast.parse(assignments_source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and not name.id.startswith("_"):
                    names.add(name.id)
    return names


def dedupe_imports_in_cell(source, covered):
    """Rewrite one code cell: drop/shrink import statements that add
    nothing new; keep the rest verbatim."""
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        segment = ast.get_source_segment(source, node) or ast.unparse(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pairs = set(import_pairs(segment))
            new_pairs = pairs - covered
            if not new_pairs:
                continue
            covered.update(new_pairs)
            if isinstance(node, ast.ImportFrom):
                segment = reemit(segment, new_pairs)
            keep.append(segment)
        else:
            keep.append(segment)
    return "\n".join(keep)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    present = [n for n in NOTEBOOKS if (SRC_DIR / f"{n}.py").exists()]
    missing = [n for n in NOTEBOOKS if n not in present]
    if missing:
        print("skipped, not written yet:", ", ".join(missing))

    covered = set()     # already-imported (module, name) pairs
    all_names = {}      # bound name -> notebook (for shadow warnings)
    merged_cells = []
    shim_kept = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in present:
            for cell in export_ipynb(name, tmp):
                if cell["cell_type"] != "code":
                    merged_cells.append(cell)
                    continue
                source = "".join(cell["source"])
                if source.strip().replace("import marimo as mo", "") == "":
                    if shim_kept:
                        continue
                    merged_cells.append(cell)
                    shim_kept = True
                    continue
                source = dedupe_imports_in_cell(source, covered)
                names = bound_names(source)
                shadow = sorted(set(all_names) & names)
                if shadow:
                    print(
                        f"note: {name} redefines "
                        f"{', '.join(shadow)} (already in another task)"
                    )
                all_names.update({n: name for n in names})
                merged_cells.append({
                    "cell_type": "code", "metadata": {},
                    "execution_count": None, "outputs": [],
                    "source": source,
                })

    target = OUT_DIR / "submission_notebook"
    ipynb = {"cells": merged_cells, "metadata": {},
             "nbformat": 4, "nbformat_minor": 5}
    (target.with_suffix(".ipynb")).write_text(json.dumps(ipynb, indent=1))

    flat = "\n\n".join(
        "".join(cell["source"]) for cell in merged_cells
        if cell["cell_type"] == "code"
    )
    (target.with_suffix(".py")).write_text(flat)
    print(f"submission notebook: {len(merged_cells)} cells "
          f"(+ the flat .py for the run check)")


def run_check():
    subprocess.run(
        ["uv", "run", "python", str(OUT_DIR / "submission_notebook.py")],
        check=True,
    )


if __name__ == "__main__":
    build()
    if "--run" in sys.argv:
        run_check()
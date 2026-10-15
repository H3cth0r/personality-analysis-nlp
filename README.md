# Project 13 — Personality Analysis

Do word habits show personality? We study the **Persona-E²** dataset:
3,111 short texts, each read by 36 people who reported how it made them
feel. Every person has Big Five scores. We count words, find topics, measure
feeling words, and compare against personality traits.

- Assignment: [`docs/Project_13_Personality_Analysis.md`](docs/Project_13_Personality_Analysis.md)

## Team

| Name | Student ID |
|---|---|
| _fill in_ | _fill in_ |
| _fill in_ | _fill in_ |
| _fill in_ | _fill in_ |

## Setup

Install `uv` once — it manages the Python version, dependencies and venv.
Pick your OS:

```bash
# macOS
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then this is the whole setup:

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, installs matching Python, creates `.venv`,
locks `uv.lock`.

## Data

Download everything (CC BY-NC-SA — research use only, never commit these
files; `data/` is gitignored):

```bash
mkdir -p data/raw/lexicons

# Persona-E2 dataset (3 files, ~8 MB)
curl -L "https://huggingface.co/datasets/CRIS-Yang/Persona-E2-Dataset/resolve/main/1_dataset_all_annotators.csv" -o data/raw/1_dataset_all_annotators.csv
curl -L "https://huggingface.co/datasets/CRIS-Yang/Persona-E2-Dataset/resolve/main/2_dataset_group_consensus.csv" -o data/raw/2_dataset_group_consensus.csv
curl -L "https://huggingface.co/datasets/CRIS-Yang/Persona-E2-Dataset/resolve/main/3_annotator_profiles.csv" -o data/raw/3_annotator_profiles.csv

# NRC VAD lexicon (the feeling-word dictionary for Task 3)
curl -L "https://raw.githubusercontent.com/Priya22/EmotionDynamics/master/lexicons/NRC_EmoLex_NRC-VAD-Lexicon.txt" -o data/raw/lexicons/NRC-VAD-Lexicon.txt
```

Check you got the right files:

```bash
ls -la data/raw/          # ~5.1 MB + ~2.6 MB + ~1.5 KB
ls -la data/raw/lexicons  # ~530 KB
```

Everything downstream regenerates from these files: notebook `00_setup`
reads them and builds `data/processed/events.parquet`.

## Notebooks

One marimo notebook per task (`00_setup` … `07_authors_method`), plain
Python cells. Run any of them with:

```bash
# live + editable — the working mode
uv run marimo edit notebooks/00_setup.py

# read-only app view (for showing around)
uv run marimo run notebooks/01_count_words.py

# static, self-contained, already executed:
uv run marimo export html notebooks/01_count_words.py -o /tmp/01.html

# plain Jupyter notebook:
uv run marimo export ipynb notebooks/01_count_words.py -o /tmp/01.ipynb
```

First-run notes, per machine:

- Data must be in `data/raw/` (see Data above). Everything else builds
  itself: the notebooks cache their outputs under `data/`
  (`events.parquet`, `speakers.parquet`, LDA sweeps) and re-use them.
- The LDA sweep (notebook `02_topics`) takes ~30 min ONCE per machine,
  run by `uv run python scripts/sweep_topics.py`. After that, seconds.
- The submission notebook (one Jupyter file with all tasks) gets
  assembled at the end of the project.

## Layout

```
docs/             brief
data/             raw/, interim/, processed/   (gitignored)
notebooks/        one marimo notebook per task (00…07)
src/personality/  the logic: loading, counting, topics, lexicon, embeddings
scripts/          build helpers (sweep runner, final assembly later)
outputs/          figures/ + tables/ — committed, the evidence
reports/          final PDF + draft
```

Notebooks: marimo (plain `.py`, merge-friendly). At submission: exported
to ONE Jupyter notebook + HTML by the merge script.
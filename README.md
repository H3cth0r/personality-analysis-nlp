# Project 13 — Personality Analysis

Do word habits show personality? We study the **Persona-E²** dataset:
3,111 short texts, each read by 36 people who reported how it made them
feel. Every person has Big Five scores. We count words, find topics, measure
feeling words, and compare against personality traits.

- Assignment: [`docs/Project_13_Personality_Analysis.md`](docs/Project_13_Personality_Analysis.md)
- Full planning: separate repo (`p-a-nlp-planning`)

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

Download (CC BY-NC-SA — research use only, never commit these files):
https://huggingface.co/datasets/CRIS-Yang/Persona-E2-Dataset

Put the 3 CSVs in `data/raw/`:

| File | Goes to |
|---|---|
| `1_dataset_all_annotators.csv` | `data/raw/` |
| `2_dataset_group_consensus.csv` | `data/raw/` |
| `3_annotator_profiles.csv` | `data/raw/` |

NRC VAD lexicon → `data/raw/lexicons/` (Priya22/EmotionDynamics on GitHub).

`data/` is gitignored. Everything regenerates from raw.

## Run

```bash
# one notebook
uv run marimo edit notebooks/01_count_words.py

# all of it: merge into one submission notebook
uv run python scripts/build_submission.py
```

## Layout

```
docs/             brief
data/             raw/, interim/, processed/   (gitignored)
notebooks/        one marimo notebook per task (00…07)
src/personality/  the logic: loading, counting, topics, lexicon, embeddings
scripts/          merge script + build helpers
outputs/          figures/ + tables/ — committed, the evidence
reports/          final PDF + draft
```

Notebooks: marimo (plain `.py`, merge-friendly). At submission: exported
to ONE Jupyter notebook + HTML by the merge script.
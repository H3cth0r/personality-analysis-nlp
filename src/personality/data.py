"""Loading and checking of the Persona-E2 data (Step 0.2)."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ANNOTATORS = range(1, 37)
EMO_COLS = [f"E{n}_emotion" for n in ANNOTATORS]
CONF_COLS = [f"E{n}_confidence" for n in ANNOTATORS]

# columns kept for events.parquet - the contract is in planning file 04
EVENT_COLS = ["id", "english_item", "data_source", "category"]


def load_raw(path=RAW_DIR / "1_dataset_all_annotators.csv"):
    """The full raw event file (3,111 rows, one per event)."""
    return pd.read_csv(path)


def load_profiles(path=RAW_DIR / "3_annotator_profiles.csv"):
    """The 36 annotator personality profiles."""
    return pd.read_csv(path)


def load_group_consensus(path=RAW_DIR / "2_dataset_group_consensus.csv"):
    """Group-level emotions per event (6 BFI-based personality groups)."""
    return pd.read_csv(path)


def build_events(raw):
    """Keep only the contract columns; wide format, one row per event."""
    missing = [c for c in EVENT_COLS + EMO_COLS + CONF_COLS if c not in raw.columns]
    if missing:
        raise KeyError(f"expected columns missing from raw CSV: {missing}")
    return raw[EVENT_COLS + EMO_COLS + CONF_COLS]


def save_events(events,
                path=PROCESSED_DIR / "events.parquet"):
    """Cache the events table for the next run."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path)


def load_events(path=PROCESSED_DIR / "events.parquet"):
    """Load the cached parquet if it exists, else rebuild from raw."""
    if path.exists():
        return pd.read_parquet(path)
    events = build_events(load_raw())
    save_events(events, path)
    return events


def check_events(raw, profiles, groups=None):
    """Quick data checks. Returns a small DataFrame for display."""
    eng = raw["english_item"].fillna("")
    emo_flat = raw[EMO_COLS].to_numpy().ravel()
    emo_labels = sorted(set(str(v) for v in emo_flat))
    conf_nulls = int(raw[CONF_COLS].isna().sum().sum())
    reactions = raw[EMO_COLS].notna().sum()
    rows = [
        ("events rows", len(raw)),
        ("profiles rows", len(profiles)),
        ("empty english_item", int((eng == "").sum())),
        ("english_item shorter than 5 chars", int((eng.str.len() < 5).sum())),
        ("emotion labels found", str(emo_labels)),
        ("distinct label count", len(emo_labels)),
        ("null confidences (total)", conf_nulls),
        ("reactions per annotator: min / median / max",
         f"{reactions.min()} / {reactions.median()} / {reactions.max()}"),
    ]
    if groups is not None:
        rows.append(("group consensus rows", len(groups)))
    return pd.DataFrame(rows, columns=["check", "value"])
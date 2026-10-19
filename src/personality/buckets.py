
"""Dominant-personality buckets (Task 4).

This module contains the code for the dominant-personality buckets (Task 4).

The dominant-personality buckets are a set of buckets that are used to group
the speakers by their dominant personality trait.

The buckets are created by the dominant_traits function, which takes a
profiles dataframe and returns a dictionary of ID -> highest-scoring Big Five trait.
"""

TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def dominant_traits(profiles):
    """Return ID -> highest-scoring Big Five trait."""
    scores = profiles.set_index("ID")[TRAIT_COLS]
    names = dict(zip(TRAIT_COLS, TRAIT_NAMES))
    return scores.rename(columns=names).idxmax(axis=1).to_dict()


def bucket_speakers(speakers, profiles):
    """Group speaker event-text piles by each speaker's dominant trait."""
    dominant = dominant_traits(profiles)
    buckets = {trait: [] for trait in TRAIT_NAMES}

    for name, pile in speakers.groupby("annotator"):
        trait = dominant[name]
        buckets[trait].extend(pile["english_item"].tolist())

    return buckets
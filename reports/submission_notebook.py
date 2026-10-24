import marimo as mo

# SETUP:
from personality import data

RAW = data.load_raw()
PROFILES = data.load_profiles()
GROUPS = data.load_group_consensus()

CHECKS = data.check_events(RAW, PROFILES, groups=GROUPS)

CHECKS

EVENTS = data.build_events(RAW)
data.save_events(EVENTS)
EVENTS.shape

# SETUP:
import pandas as pd
import matplotlib.pyplot as plt
from personality import counting

SPEAKERS = data.load_speakers()
PROFILES = data.load_profiles()

T1_ROWS = {
    name: counting.speaker_stats(pile["english_item"])
    for name, pile in SPEAKERS.groupby("annotator")
}
# rows become speakers; drop the unshown set column; keep whole numbers
T1_STATS = pd.DataFrame(T1_ROWS).T.drop(columns="vocabulary_set").astype(int)
T1_TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
T1_TRAIT_NAMES = ["openness", "conscientiousness", "extraversion",
               "agreeableness", "neuroticism"]
T1_TRAITS = (PROFILES.set_index("ID")[T1_TRAIT_COLS]
          .rename(columns=dict(zip(T1_TRAIT_COLS, T1_TRAIT_NAMES))))
T1_JOINED = T1_STATS.join(T1_TRAITS)
T1_STATS.to_csv("outputs/tables/task1_speaker_stats.csv")

T1_JOINED

from matplotlib.ticker import MaxNLocator

T1_FIG, T1_AXES = plt.subplots(6, 6, figsize=(17, 14), sharex=False)
for _ax1, (_name, _pile) in zip(T1_AXES.flat, SPEAKERS.groupby("annotator")):
    _series = counting.chunked_series(_pile["english_item"], chunks=10)
    _ax1.plot(_series["tokens"], _series["repetitions"], label="repetitions")
    _ax1.plot(_series["tokens"], _series["no_words"], label="negations")
    _ax1.plot(_series["tokens"], _series["yes_words"], label="confirmations")
    _ax1.set_title(f"{_name} ({int(_pile.shape[0])} texts)", fontsize=9)
    _ax1.tick_params(labelsize=7)
    _ax1.xaxis.set_major_locator(MaxNLocator(4))
    _ax1.yaxis.set_major_locator(MaxNLocator(5))
# wipe the panel labels (the shared supxlabel carries them)
for _ax2 in T1_AXES.flat:
    _ax2.set_xlabel("")
    _ax2.set_ylabel("")
T1_AXES[0, 0].legend(fontsize=8)
T1_FIG.supxlabel("tokens so far")
T1_FIG.tight_layout()
T1_FIG.savefig("outputs/figures/task1_evolution.png", dpi=150)

T1_FIG

T1_CORR = (T1_JOINED[["tokens", "vocabulary_size", "repetitions", "yes_words",
                "no_words"] + T1_TRAIT_NAMES]
        .corr(method="spearman")
        .loc[T1_TRAIT_NAMES, ["tokens", "vocabulary_size", "repetitions",
                           "yes_words", "no_words"]])
T1_CORR.round(3).to_csv("outputs/tables/task1_trait_correlations.csv")
T1_CORR.round(2)

# SETUP:
from personality import topics

SPEAKERS = data.load_speakers()
PROFILES = data.load_profiles()

T2_FIG, (T2_AX1, T2_AX2) = plt.subplots(1, 2, figsize=(13, 5))
for _, pile_b in SPEAKERS.groupby("annotator"):
    growth = counting.vocab_growth(pile_b["english_item"])
    T2_AX1.plot(growth["tokens"], growth["vocab_size"], linewidth=1)
    # x as a share of the pile (0 to 1): the shape, not the length
    frac = growth["tokens"] / growth["tokens"].iloc[-1]
    T2_AX2.plot(frac, growth["vocab_size"], linewidth=1)
T2_AX1.set_title("raw tokens", fontsize=10)
T2_AX1.set_xlabel("tokens so far")
T2_AX1.set_ylabel("vocabulary size")
T2_AX2.set_title("fraction of pile", fontsize=10)
T2_AX2.set_xlabel("fraction of the speaker's texts read")
T2_AX2.set_ylabel("vocabulary size")
T2_FIG.tight_layout()
T2_FIG.savefig("outputs/figures/task2_vocab_growth.png", dpi=150)

T2_FIG

T2_SWEEPS = {}
T2_FIG2, T2_AXES = plt.subplots(6, 6, figsize=(17, 14))
for ax_b, (name_b, pile_c) in zip(T2_AXES.flat, SPEAKERS.groupby("annotator")):
    result = topics.sweep_speaker(name_b, pile_c["english_item"])
    T2_SWEEPS[name_b] = result
    # the right axis = the second scale: gray = perplexity
    ax2 = ax_b.twinx()
    l1 = ax_b.plot(result["k"], result["coherence"], "o-", label="coherence")
    l2 = ax2.plot(result["k"], result["perplexity"], "s--",
                  color="gray", label="perplexity")
    ax_b.set_title(name_b, fontsize=9)
    ax_b.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
T2_AXES[0, 0].legend(l1 + l2, ["coherence", "perplexity"], fontsize=8)
T2_FIG2.supxlabel("number of topics (k)")
T2_FIG2.tight_layout()
T2_FIG2.savefig("outputs/figures/task2_lda_sweeps.png", dpi=150)

T2_FIG2

T2_BEST = pd.DataFrame(
    {"best_k": {name2: topics.best_k(r) for name2, r in T2_SWEEPS.items()},
     "coherence": {name2: r["coherence"].max() for name2, r in T2_SWEEPS.items()}}
)
T2_TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
T2_TRAIT_NAMES = ["openness", "conscientiousness", "extraversion",
               "agreeableness", "neuroticism"]
T2_TRAITS = (PROFILES.set_index("ID")[T2_TRAIT_COLS]
          .rename(columns=dict(zip(T2_TRAIT_COLS, T2_TRAIT_NAMES))))
T2_JOINED = T2_BEST.join(T2_TRAITS)
T2_CORR = (T2_JOINED.corr(method="spearman")
        .loc[T2_TRAIT_NAMES, ["best_k", "coherence"]])
T2_JOINED.to_csv("outputs/tables/task2_best_topics.csv")
T2_CORR.round(3).to_csv("outputs/tables/task2_trait_correlations.csv")

T2_BEST

T2_CORR

# SETUP:
from personality import vad

SPEAKERS = data.load_speakers()
PROFILES = data.load_profiles()
VAD = vad.load_vad()

T3_ROWS = {
    name: vad.speaker_vad(pile["english_item"], VAD)
    for name, pile in SPEAKERS.groupby("annotator")
}
T3_STATS = pd.DataFrame(T3_ROWS).T
T3_STATS["n_vad_hits"] = T3_STATS["n_vad_hits"].astype(int)
T3_TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
T3_TRAIT_NAMES = ["openness", "conscientiousness", "extraversion",
               "agreeableness", "neuroticism"]
T3_TRAITS = (PROFILES.set_index("ID")[T3_TRAIT_COLS]
          .rename(columns=dict(zip(T3_TRAIT_COLS, T3_TRAIT_NAMES))))
T3_JOINED = T3_STATS.join(T3_TRAITS)
T3_STATS.to_csv("outputs/tables/task3_speaker_vad.csv")

T3_JOINED

T3_ORDER = [f"E{n}" for n in range(1, 37)]
T3_PLOT = T3_STATS.reindex(T3_ORDER)
T3_FIG, T3_AX = plt.subplots(figsize=(14, 5))
T3_AX.plot(T3_PLOT.index, T3_PLOT["valence_mean"], "o-",
           label="valence", markersize=4)
T3_AX.plot(T3_PLOT.index, T3_PLOT["arousal_mean"], "s-",
           label="arousal", markersize=4)
T3_AX.plot(T3_PLOT.index, T3_PLOT["dominance_mean"], "^-",
           label="dominance", markersize=4)
T3_AX.set_ylabel("mean NRC score (hits only)")
T3_AX.set_xlabel("speaker")
T3_AX.tick_params(axis="x", labelsize=8, rotation=90)
T3_AX.legend()
T3_FIG.tight_layout()
T3_FIG.savefig("outputs/figures/task3_vad_means.png", dpi=150)

T3_FIG

T3_MEAN_COLS = ["valence_mean", "arousal_mean", "dominance_mean"]
T3_CORR = (T3_JOINED[T3_MEAN_COLS + T3_TRAIT_NAMES]
        .corr(method="spearman")
        .loc[T3_TRAIT_NAMES, T3_MEAN_COLS])
T3_CORR.round(3).to_csv("outputs/tables/task3_trait_correlations.csv")
T3_CORR.round(2)

# SETUP:

from personality import buckets

SPEAKERS = data.load_speakers()
PROFILES = data.load_profiles()

T4_DOMINANT = buckets.dominant_traits(PROFILES)

T4_DOMINANT_COUNTS = (
    pd.Series(T4_DOMINANT)
    .value_counts()
    .reindex(buckets.TRAIT_NAMES, fill_value=0)
)

T4_DOMINANT_COUNTS

T4_BUCKETS = buckets.bucket_speakers(SPEAKERS, PROFILES)

# rows become buckets; drop the unshown set column; keep whole numbers
T4_STATS = pd.DataFrame({
    trait: counting.speaker_stats(texts)
    for trait, texts in T4_BUCKETS.items()
}).T.drop(columns="vocabulary_set").astype(int)

T4_STATS

T4_FIG, T4_AXES = plt.subplots(1, 5, figsize=(18, 4), sharex=False)

for _ax, _trait in zip(T4_AXES, buckets.TRAIT_NAMES):
    _series = counting.chunked_series(T4_BUCKETS[_trait], chunks=10)

    _ax.plot(_series["tokens"], _series["repetitions"], label="repetitions")
    _ax.plot(_series["tokens"], _series["no_words"], label="negations")
    _ax.plot(_series["tokens"], _series["yes_words"], label="confirmations")

    _ax.set_title(_trait)
    _ax.tick_params(labelsize=8)

T4_AXES[0].set_ylabel("count")
T4_FIG.supxlabel("tokens so far")
T4_AXES[0].legend(fontsize=8)

T4_FIG.tight_layout()

T4_FIG

T4_STATS.to_csv("outputs/tables/task4_bucket_stats.csv")

T4_FIG.savefig(
    "outputs/figures/task4_bucket_evolution.png",
    dpi=150,
)

T4_NORMALIZED = T4_STATS.div(T4_STATS["tokens"], axis=0) * 1000
T4_NORMALIZED[["repetitions", "yes_words", "no_words"]]

# SETUP:
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from personality import embeddings

T5_SPEAKERS = data.load_speakers()
T5_PROFILES = data.load_profiles()

# Reference labels: each annotator's dominant personality trait.
T5_ACTUAL_TRAITS = buckets.dominant_traits(T5_PROFILES)

T5_ACTUAL = pd.Series(
    T5_ACTUAL_TRAITS,
    name="actual_trait",
    dtype="object",
)
T5_ACTUAL.index.name = "annotator"

# Use every valid, non-empty speaker text.
T5_VALID_SPEAKERS = T5_SPEAKERS.dropna(
    subset=["annotator", "english_item"]
).copy()

# force plain strings (some cells arrive as gaps); trim stray spaces
T5_VALID_SPEAKERS["annotator"] = (
    T5_VALID_SPEAKERS["annotator"].astype(str).str.strip()
)
T5_VALID_SPEAKERS["english_item"] = (
    T5_VALID_SPEAKERS["english_item"].astype(str).str.strip()
)

# drop rows with an empty annotator or empty text; renumber after
T5_VALID_SPEAKERS = T5_VALID_SPEAKERS[
    (T5_VALID_SPEAKERS["annotator"] != "")
    & (T5_VALID_SPEAKERS["english_item"] != "")
].reset_index(drop=True)

# show the row count at each step: the cut's cost
T5_DATA_SUMMARY = pd.DataFrame({
    "measure": [
        "All speaker rows",
        "Valid non-empty text rows",
        "Annotators with reference labels",
    ],
    "count": [
        len(T5_SPEAKERS),
        len(T5_VALID_SPEAKERS),
        len(T5_ACTUAL),
    ],
})

T5_DATA_SUMMARY

from pathlib import Path

T5_SCORES_CACHE = Path("data/interim/task5_speaker_scores.parquet")
T5_COVERAGE_CACHE = Path("data/interim/task5_term_coverage.parquet")

# same numbers every run: load the saved scores, or train once and save
if T5_SCORES_CACHE.exists():
    T5_SPEAKER_SCORES = pd.read_parquet(T5_SCORES_CACHE)
    T5_TERM_COVERAGE = pd.read_parquet(T5_COVERAGE_CACHE)
    T5_TRAINING_SUMMARY = {"loaded": "from data/interim caches"}
else:
    T5_DEFINITION_TERMS = [
        term
        for terms in embeddings.TRAIT_TERMS.values()
        for term in terms
    ]

    T5_TRAIN_TEXTS = (
        T5_VALID_SPEAKERS["english_item"].tolist()
        + T5_DEFINITION_TERMS
    )

    T5_MODEL = embeddings.train_doc2vec(
        T5_TRAIN_TEXTS,
        vector_size=50,
        epochs=10,
        seed=42,
    )

    T5_TRAINING_SUMMARY = {
        "valid speaker text rows": len(T5_VALID_SPEAKERS),
        "definition terms added": len(T5_DEFINITION_TERMS),
        "training documents": len(T5_TRAIN_TEXTS),
        "vocabulary size": len(T5_MODEL.wv),
        "vector size": T5_MODEL.vector_size,
        "epochs": T5_MODEL.epochs,
    }

    T5_TERM_VECTORS = embeddings.score_traits(
        T5_MODEL,
        embeddings.TRAIT_TERMS,
    )

    T5_SPEAKER_TEXTS = (
        T5_VALID_SPEAKERS
        .groupby("annotator", sort=True)["english_item"]
        .agg(" ".join)
    )

    T5_SPEAKER_SCORES = embeddings.predict_traits(
        T5_MODEL,
        T5_SPEAKER_TEXTS.tolist(),
        T5_TERM_VECTORS,
    )
    T5_SPEAKER_SCORES.index = T5_SPEAKER_TEXTS.index

    T5_COVERAGE_ROWS = []
    for _trait, _terms in embeddings.TRAIT_TERMS.items():
        for _term in _terms:
            words = embeddings._tokens(_term)
            known_words = [
                word for word in words if word in T5_MODEL.wv
            ]
            T5_COVERAGE_ROWS.append({
                "trait": _trait,
                "definition_term": _term,
                "token_count": len(words),
                "known_token_count": len(known_words),
                "known_tokens": ", ".join(known_words),
                "covered": bool(known_words),
            })
    T5_TERM_COVERAGE = pd.DataFrame(T5_COVERAGE_ROWS)

    T5_SPEAKER_SCORES.to_parquet(T5_SCORES_CACHE)
    T5_TERM_COVERAGE.to_parquet(T5_COVERAGE_CACHE)

T5_TRAINING_SUMMARY

T5_PREDICTED = embeddings.predict_labels(T5_SPEAKER_SCORES)

T5_RESULTS = pd.DataFrame({
    "actual_trait": T5_ACTUAL,
    "predicted_trait": T5_PREDICTED,
})

T5_RESULTS = T5_RESULTS.join(
    T5_SPEAKER_SCORES.add_prefix("similarity_"),
    how="left",
)
T5_RESULTS.index.name = "annotator"

T5_RESULTS

T5_BFI_COLUMNS = {
    "openness": "Open.",
    "conscientiousness": "Cons.",
    "extraversion": "Extra.",
    "agreeableness": "Agree.",
    "neuroticism": "Neuro.",
}

T5_BFI = T5_PROFILES.set_index("ID")
T5_ALIGNED_BFI = T5_BFI.reindex(T5_SPEAKER_SCORES.index)

T5_CORRELATION_ROWS = []

for T5_TRAIT, T5_BFI_COLUMN in T5_BFI_COLUMNS.items():
    T5_COMPARISON = pd.DataFrame({
        "similarity": T5_SPEAKER_SCORES[T5_TRAIT],
        "bfi_score": T5_ALIGNED_BFI[T5_BFI_COLUMN],
    }).dropna()

    T5_RHO = T5_COMPARISON["similarity"].corr(
        T5_COMPARISON["bfi_score"],
        method="spearman",
    )

    T5_CORRELATION_ROWS.append({
        "trait": T5_TRAIT,
        "bfi_column": T5_BFI_COLUMN,
        "n": len(T5_COMPARISON),
        "spearman_rho": T5_RHO,
    })

T5_SPEARMAN = pd.DataFrame(T5_CORRELATION_ROWS)

T5_SPEARMAN

T5_TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

T5_EVALUATION = T5_RESULTS.dropna(
    subset=["actual_trait", "predicted_trait"]
)

T5_Y_TRUE = T5_EVALUATION["actual_trait"]
T5_Y_PRED = T5_EVALUATION["predicted_trait"]

T5_ACCURACY = accuracy_score(T5_Y_TRUE, T5_Y_PRED)
T5_MACRO_F1 = f1_score(
    T5_Y_TRUE,
    T5_Y_PRED,
    labels=T5_TRAIT_NAMES,
    average="macro",
    zero_division=0,
)

T5_MAJORITY_BASELINE = (
    T5_Y_TRUE.value_counts(normalize=True).max()
    if len(T5_Y_TRUE)
    else float("nan")
)

T5_UNIFORM_BASELINE = 1 / len(T5_TRAIT_NAMES)

T5_METRICS = pd.DataFrame({
    "metric": [
        "evaluated_annotators",
        "accuracy",
        "macro_f1",
        "uniform_random_baseline",
        "majority_class_baseline",
    ],
    "value": [
        len(T5_EVALUATION),
        T5_ACCURACY,
        T5_MACRO_F1,
        T5_UNIFORM_BASELINE,
        T5_MAJORITY_BASELINE,
    ],
})

T5_CONFUSION = pd.DataFrame(
    confusion_matrix(
        T5_Y_TRUE,
        T5_Y_PRED,
        labels=T5_TRAIT_NAMES,
    ),
    index=T5_TRAIT_NAMES,
    columns=T5_TRAIT_NAMES,
)
T5_CONFUSION.index.name = "actual_trait"
T5_CONFUSION.columns.name = "predicted_trait"

T5_REPORT = pd.DataFrame(
    classification_report(
        T5_Y_TRUE,
        T5_Y_PRED,
        labels=T5_TRAIT_NAMES,
        output_dict=True,
        zero_division=0,
    )
).T

T5_METRICS

T5_MEAN_SIMILARITY = pd.DataFrame({
    "trait": T5_TRAIT_NAMES,
    "mean_similarity": [
        T5_SPEAKER_SCORES[trait].mean()
        for trait in T5_TRAIT_NAMES
    ],
    "std_similarity": [
        T5_SPEAKER_SCORES[trait].std()
        for trait in T5_TRAIT_NAMES
    ],
})

T5_MEAN_SIMILARITY

T5_SIM_FIG, T5_SIM_AX = plt.subplots(figsize=(9, 5))

T5_SIM_AX.bar(
    T5_MEAN_SIMILARITY["trait"],
    T5_MEAN_SIMILARITY["mean_similarity"],
)
T5_SIM_AX.set_title("Mean similarity by personality trait")
T5_SIM_AX.set_xlabel("Personality trait")
T5_SIM_AX.set_ylabel("Mean cosine similarity")
T5_SIM_AX.tick_params(axis="x", rotation=25)

T5_SIM_FIG.tight_layout()

T5_SIM_FIG

T5_CM_FIG, T5_CM_AX = plt.subplots(figsize=(8, 6))

T5_CM_AX.imshow(T5_CONFUSION.values)
T5_CM_AX.set_xticks(range(len(T5_CONFUSION.columns)))
T5_CM_AX.set_xticklabels(T5_CONFUSION.columns, rotation=35, ha="right")
T5_CM_AX.set_yticks(range(len(T5_CONFUSION.index)))
T5_CM_AX.set_yticklabels(T5_CONFUSION.index)

for row in range(len(T5_CONFUSION.index)):
    for col in range(len(T5_CONFUSION.columns)):
        T5_CM_AX.text(
            col,
            row,
            T5_CONFUSION.iloc[row, col],
            ha="center",
            va="center",
        )

T5_CM_AX.set_xlabel("Predicted trait")
T5_CM_AX.set_ylabel("Actual trait")
T5_CM_AX.set_title("Personality trait confusion matrix")

T5_CM_FIG.tight_layout()

T5_CM_FIG

T5_COVERAGE_SUMMARY = (
    T5_TERM_COVERAGE.groupby("trait", as_index=False)
    .agg(
        total_terms=("definition_term", "count"),
        covered_terms=("covered", "sum"),
    )
)

T5_COVERAGE_SUMMARY["coverage_percent"] = (
    100
    * T5_COVERAGE_SUMMARY["covered_terms"]
    / T5_COVERAGE_SUMMARY["total_terms"]
)

T5_TERM_COVERAGE

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)

T5_RESULTS.to_csv("outputs/tables/task5_predictions.csv")
T5_METRICS.to_csv("outputs/tables/task5_metrics.csv", index=False)
T5_CONFUSION.to_csv("outputs/tables/task5_confusion_matrix.csv")
T5_REPORT.to_csv("outputs/tables/task5_classification_report.csv")
T5_MEAN_SIMILARITY.to_csv(
    "outputs/tables/task5_mean_similarity.csv", index=False
)
T5_SPEAKER_SCORES.to_csv(
    "outputs/tables/task5_speaker_similarity.csv"
)
T5_SPEARMAN.to_csv(
"outputs/tables/task5_spearman_correlations.csv",
index=False,
)
T5_TERM_COVERAGE.to_csv(
    "outputs/tables/task5_term_coverage.csv", index=False
)
T5_COVERAGE_SUMMARY.to_csv(
    "outputs/tables/task5_term_coverage_summary.csv", index=False
)

T5_SIM_FIG.savefig(
    "outputs/figures/task5_mean_similarity.png",
    dpi=150,
    bbox_inches="tight",
)
T5_CM_FIG.savefig(
    "outputs/figures/task5_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)

"Task 5 outputs saved."

# SETUP:
import hashlib
import json


from personality.llm import PROMPT_VERSION, explain_trait

T6_PROFILES = data.load_profiles()
T6_SPEAKERS = data.load_speakers()

T6_TASK1 = pd.read_csv(
    "outputs/tables/task1_speaker_stats.csv",
    index_col=0,
)
T6_TASK3 = pd.read_csv(
    "outputs/tables/task3_speaker_vad.csv",
    index_col=0,
)
T6_TASK4 = pd.read_csv(
    "outputs/tables/task4_bucket_stats.csv",
    index_col=0,
)
T6_PREDICTIONS = pd.read_csv(
    "outputs/tables/task5_predictions.csv"
)
T6_SIMILARITY = pd.read_csv(
    "outputs/tables/task5_speaker_similarity.csv",
    index_col=0,
)

# Join the existing measurements by anonymized annotator ID.
# the CSV indexes arrive with stray spaces: make them plain strings
for T6_FRAME in (
    T6_TASK1,
    T6_TASK3,
    T6_TASK4,
    T6_SIMILARITY,
):
    T6_FRAME.index = T6_FRAME.index.astype(str).str.strip()

T6_PREDICTION_DATA = T6_PREDICTIONS.set_index("annotator")
T6_PREDICTION_DATA.index = (
    T6_PREDICTION_DATA.index.astype(str).str.strip()
)

T6_TRAITS = list(buckets.TRAIT_NAMES)
T6_SIM_COLUMNS = [f"similarity_{trait}" for trait in T6_TRAITS]

T6_ACTUAL = pd.Series(
    buckets.dominant_traits(T6_PROFILES),
    name="actual_trait",
)
T6_ACTUAL.index = T6_ACTUAL.index.astype(str).str.strip()

T6_PROFILE_RENAME = {
    "Open.": "profile_openness",
    "Cons.": "profile_conscientiousness",
    "Extra.": "profile_extraversion",
    "Agree.": "profile_agreeableness",
    "Neuro.": "profile_neuroticism",
}
T6_BFI = (
    T6_PROFILES.set_index("ID")[list(T6_PROFILE_RENAME)]
    .rename(columns=T6_PROFILE_RENAME)
)
T6_BFI.index = T6_BFI.index.astype(str).str.strip()

T6_CASES = (
    T6_PREDICTION_DATA[["actual_trait", "predicted_trait"]]
    .join(T6_SIMILARITY.add_prefix("similarity_"), how="inner")
    .join(T6_TASK1, how="left")
    .join(T6_TASK3, how="left")
    .join(T6_BFI, how="left")
)

T6_CASES["prediction_correct"] = (
    T6_CASES["actual_trait"] == T6_CASES["predicted_trait"]
)

# rank the five trait scores; keep the top two and the gap
def T6_score_summary(row):
    T6_RANKED = row[T6_SIM_COLUMNS].astype(float).sort_values(
        ascending=False
    )
    return pd.Series({
        "top_similarity_trait": T6_RANKED.index[0].replace(
            "similarity_", ""
        ),
        "top_similarity": T6_RANKED.iloc[0],
        "second_similarity": T6_RANKED.iloc[1],
        "score_margin": T6_RANKED.iloc[0] - T6_RANKED.iloc[1],
    })

T6_CASES = T6_CASES.join(
    T6_CASES.apply(T6_score_summary, axis=1)
)

# Normalize count measures so buckets of different sizes are comparable.
T6_BUCKET_RATES = T6_TASK4.copy()
T6_BUCKET_RATES["repetitions_per_1000"] = (
    1000 * T6_BUCKET_RATES["repetitions"] / T6_BUCKET_RATES["tokens"]
)
T6_BUCKET_RATES["confirmations_per_1000"] = (
    1000 * T6_BUCKET_RATES["yes_words"] / T6_BUCKET_RATES["tokens"]
)
T6_BUCKET_RATES["negations_per_1000"] = (
    1000 * T6_BUCKET_RATES["no_words"] / T6_BUCKET_RATES["tokens"]
)
T6_BUCKET_RATES = T6_BUCKET_RATES.reindex(T6_TRAITS)

# Select one case per actual trait. Prefer a wrong prediction,
# then a larger top-versus-second similarity margin.
T6_SELECTED_IDS = []
for T6_TRAIT in T6_TRAITS:
    T6_GROUP = T6_CASES[T6_CASES["actual_trait"] == T6_TRAIT]
    if T6_GROUP.empty:
        continue

    T6_WRONG = T6_GROUP[~T6_GROUP["prediction_correct"]]
    T6_CANDIDATES = T6_WRONG if not T6_WRONG.empty else T6_GROUP
    T6_CHOSEN = T6_CANDIDATES.sort_values(
        "score_margin", ascending=False
    ).head(1)
    T6_SELECTED_IDS.extend(T6_CHOSEN.index.tolist())

T6_SELECTED = T6_CASES.loc[T6_SELECTED_IDS].copy()

T6_SELECTED[
    [
        "actual_trait", "predicted_trait",
        "top_similarity_trait", "top_similarity",
        "second_similarity", "score_margin",
        "prediction_correct",
    ]
].round(3).reset_index()

def T6_number(value):
    return round(float(value), 4) if pd.notna(value) else None

def T6_group_rates(trait):
    if trait not in T6_BUCKET_RATES.index:
        return {}
    return {
        name: T6_number(T6_BUCKET_RATES.loc[trait, name])
        for name in (
            "repetitions_per_1000",
            "confirmations_per_1000",
            "negations_per_1000",
        )
    }

# build the exact number dict the LLM will see, one per case
T6_EVIDENCE_BY_ID = {}

for T6_EVIDENCE_ANNOTATOR, T6_ROW in T6_SELECTED.iterrows():
    T6_PROFILE = {
        trait: T6_number(T6_ROW.get(f"profile_{trait}"))
        for trait in T6_TRAITS
    }
    T6_SIMILARITIES = {
        trait: T6_number(T6_ROW[f"similarity_{trait}"])
        for trait in T6_TRAITS
    }

    T6_PILE = T6_SPEAKERS.loc[
        T6_SPEAKERS["annotator"].astype(str) == str(T6_EVIDENCE_ANNOTATOR),
        "english_item",
    ]

    T6_EXAMPLES = []
    for T6_TEXT in T6_PILE.dropna().astype(str).drop_duplicates():
        T6_TEXT = " ".join(T6_TEXT.split())
        if T6_TEXT:
            T6_EXAMPLES.append(T6_TEXT[:220])
        if len(T6_EXAMPLES) == 2:
            break

    T6_EVIDENCE_BY_ID[str(T6_EVIDENCE_ANNOTATOR)] = {
        "annotator_id": str(T6_EVIDENCE_ANNOTATOR),
        "actual_dominant_trait": T6_ROW["actual_trait"],
        "task5_predicted_trait": T6_ROW["predicted_trait"],
        "highest_similarity_trait": T6_ROW["top_similarity_trait"],
        "top_similarity": T6_number(T6_ROW["top_similarity"]),
        "second_highest_similarity": T6_number(
            T6_ROW["second_similarity"]
        ),
        "top_score_margin": T6_number(T6_ROW["score_margin"]),
        "similarity_scores_not_probabilities": T6_SIMILARITIES,
        "big_five_profile_scores": T6_PROFILE,
        "task1_word_statistics": {
            "tokens": T6_number(T6_ROW.get("tokens")),
            "vocabulary_size": T6_number(T6_ROW.get("vocabulary_size")),
            "repetitions": T6_number(T6_ROW.get("repetitions")),
            "repetitions_per_1000": T6_number(
                1000 * T6_ROW["repetitions"] / T6_ROW["tokens"]
            ),
            "confirmations_per_1000": T6_number(
                1000 * T6_ROW["yes_words"] / T6_ROW["tokens"]
            ),
            "negations_per_1000": T6_number(
                1000 * T6_ROW["no_words"] / T6_ROW["tokens"]
            ),
        },
        "task3_vad_statistics": {
            name: T6_number(T6_ROW.get(name))
            for name in (
                "valence_mean", "valence_std",
                "arousal_mean", "arousal_std",
                "dominance_mean", "dominance_std",
            )
        },
        "task4_bucket_rates_per_1000_tokens": {
            "actual_trait_group": T6_group_rates(
                T6_ROW["actual_trait"]
            ),
            "predicted_trait_group": T6_group_rates(
                T6_ROW["predicted_trait"]
            ),
            "all_trait_groups": {
                trait: T6_group_rates(trait)
                for trait in T6_TRAITS
            },
        },
        "event_text_examples_not_authored_by_annotator": T6_EXAMPLES,
    }

T6_EVIDENCE_TABLE = T6_SELECTED[[
    "actual_trait",
    "predicted_trait",
    "top_similarity_trait",
    "top_similarity",
    "second_similarity",
    "score_margin",
    "prediction_correct",
    "tokens",
    "vocabulary_size",
    "valence_mean",
    "arousal_mean",
    "dominance_mean",
]].copy()

# Add per-annotator count rates (per 1,000 tokens) from Task 1.
T6_EVIDENCE_TABLE["repetitions_per_1000"] = (
    1000 * T6_SELECTED["repetitions"] / T6_SELECTED["tokens"]
)
T6_EVIDENCE_TABLE["confirmations_per_1000"] = (
    1000 * T6_SELECTED["yes_words"] / T6_SELECTED["tokens"]
)
T6_EVIDENCE_TABLE["negations_per_1000"] = (
    1000 * T6_SELECTED["no_words"] / T6_SELECTED["tokens"]
)
T6_EVIDENCE_TABLE = T6_EVIDENCE_TABLE.round(3).reset_index()

T6_EVIDENCE_TABLE

# Create the output directory next to the file write.
T6_OUTPUT_TABLES = Path("outputs/tables")
T6_OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
T6_CACHE_PATH = T6_OUTPUT_TABLES / "task6_llm_explanations.csv"

# Cache successful responses so reruns do not repeat API requests.
T6_CACHE_BY_ID = {}

if T6_CACHE_PATH.exists():
    T6_OLD_CACHE = pd.read_csv(T6_CACHE_PATH).fillna("")
    T6_CACHE_BY_ID = {
        str(row["annotator"]): row.to_dict()
        for _, row in T6_OLD_CACHE.iterrows()
    }

T6_CURRENT_RESULTS = []

for T6_ANNOTATOR, T6_EVIDENCE in T6_EVIDENCE_BY_ID.items():
    T6_EVIDENCE_HASH = hashlib.sha256(
        json.dumps(
            T6_EVIDENCE,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    T6_OLD = T6_CACHE_BY_ID.get(T6_ANNOTATOR, {})
    T6_CACHE_VALID = (
        T6_OLD.get("prompt_version") == PROMPT_VERSION
        and T6_OLD.get("evidence_hash") == T6_EVIDENCE_HASH
        and T6_OLD.get("status") == "success"
        and bool(T6_OLD.get("explanation"))
    )

    if T6_CACHE_VALID:
        # Reuse the explanation and update its word count.
        T6_RESULT = T6_OLD.copy()
        T6_RESULT["word_count"] = len(
            str(T6_RESULT["explanation"]).split()
        )
        T6_CACHE_BY_ID[T6_ANNOTATOR] = T6_RESULT

    else:
        try:
            T6_RESPONSE = explain_trait(T6_EVIDENCE)
            T6_EXPLANATION = T6_RESPONSE["explanation"]

            T6_RESULT = {
                "annotator": T6_ANNOTATOR,
                "actual_trait": T6_EVIDENCE["actual_dominant_trait"],
                "predicted_trait": T6_EVIDENCE["task5_predicted_trait"],
                "score_margin": T6_EVIDENCE["top_score_margin"],
                "model": T6_RESPONSE["model"],
                "prompt_version": T6_RESPONSE["prompt_version"],
                "evidence_hash": T6_EVIDENCE_HASH,
                "explanation": T6_EXPLANATION,
                "word_count": len(T6_EXPLANATION.split()),
                "status": "success",
                "error": "",
            }

            # Save each successful response immediately.
            T6_CACHE_BY_ID[T6_ANNOTATOR] = T6_RESULT
            pd.DataFrame(T6_CACHE_BY_ID.values()).to_csv(
                T6_CACHE_PATH,
                index=False,
            )

        except RuntimeError as T6_ERROR:
            T6_RESULT = {
                "annotator": T6_ANNOTATOR,
                "actual_trait": T6_EVIDENCE["actual_dominant_trait"],
                "predicted_trait": T6_EVIDENCE["task5_predicted_trait"],
                "score_margin": T6_EVIDENCE["top_score_margin"],
                "model": "",
                "prompt_version": PROMPT_VERSION,
                "evidence_hash": T6_EVIDENCE_HASH,
                "explanation": "",
                "word_count": 0,
                "status": "error",
                "error": str(T6_ERROR),
            }

    T6_CURRENT_RESULTS.append(T6_RESULT)

# Persist word counts even if all explanations came from the cache.
pd.DataFrame(T6_CACHE_BY_ID.values()).to_csv(
    T6_CACHE_PATH,
    index=False,
)

T6_EXPLANATIONS = pd.DataFrame(T6_CURRENT_RESULTS)

T6_EXPLANATIONS[
    [
        "annotator",
        "actual_trait",
        "predicted_trait",
        "score_margin",
        "model",
        "prompt_version",
        "evidence_hash",
        "word_count",
        "status",
        "explanation",
        "error",
    ]
]

T6_SUCCEEDED = (
    int((T6_EXPLANATIONS["status"] == "success").sum())
    if not T6_EXPLANATIONS.empty else 0
)

mo.md(f"""
**LLM generation status:** {T6_SUCCEEDED} of
{len(T6_EXPLANATIONS)} selected explanations were generated successfully.

**Word-count check:** The prompt requests no more than 250 words.
The saved counts are E3: 284, E9: 217, E10: 225, E17: 233, and
E22: 217 words. E3 exceeds the requested limit. Counts are recorded
in the CSV, not enforced; longer responses are kept.

**What the LLM explanations show**

The explanations use the supplied similarity scores, Big Five
profiles, word statistics and VAD measurements to discuss supporting
evidence, counter-arguments and alternative interpretations.

All five cases received neuroticism as the predicted trait. Only E17
has neuroticism as its actual dominant trait.

The explanations provide a structured interpretation, but their
claims must be checked against the supplied measurements. A plausible
narrative is not evidence that a prediction is correct.

These explanations come after the prediction: they describe the
result, not what caused it. The LLM cannot inspect Doc2Vec's internal
computations or establish the causal reason for an individual prediction.
""")

T6_HEAT_FIG, T6_HEAT_AX = plt.subplots(
    figsize=(9, max(3.5, 0.7 * len(T6_SELECTED)))
)
T6_HEAT_DATA = T6_SELECTED[T6_SIM_COLUMNS].astype(float).to_numpy()

T6_HEAT_IMAGE = T6_HEAT_AX.imshow(
    T6_HEAT_DATA, aspect="auto"
)
T6_HEAT_AX.set_xticks(range(len(T6_TRAITS)))
T6_HEAT_AX.set_xticklabels(T6_TRAITS, rotation=30, ha="right")
T6_HEAT_AX.set_yticks(range(len(T6_SELECTED)))
T6_HEAT_AX.set_yticklabels([
    f"{annotator}: {row.actual_trait} → {row.predicted_trait}"
    for annotator, row in T6_SELECTED.iterrows()
])
T6_HEAT_AX.set_title("Task 5 trait similarities for selected cases")
T6_HEAT_FIG.colorbar(
    T6_HEAT_IMAGE, ax=T6_HEAT_AX, label="Cosine similarity"
)
T6_HEAT_FIG.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
T6_HEAT_FIG.savefig(
    Path("outputs/figures") / "task6_similarity_heatmap.png",
    dpi=150, bbox_inches="tight",
)

T6_HEAT_FIG

T6_RATE_COLUMNS = [
    ("repetitions_per_1000", "Repetitions per 1,000 tokens"),
    ("confirmations_per_1000", "Confirmations per 1,000 tokens"),
    ("negations_per_1000", "Negations per 1,000 tokens"),
]

T6_RATE_FIG, T6_RATE_AXES = plt.subplots(1, 3, figsize=(14, 4))

for T6_AX, (T6_COLUMN, T6_TITLE) in zip(
    T6_RATE_AXES, T6_RATE_COLUMNS
):
    T6_AX.bar(
        T6_BUCKET_RATES.index, T6_BUCKET_RATES[T6_COLUMN]
    )
    T6_AX.set_title(T6_TITLE)
    T6_AX.tick_params(axis="x", rotation=70)
    T6_AX.set_ylabel("Count per 1,000 tokens")

T6_RATE_FIG.suptitle("Task 4 personality-bucket word rates")
T6_RATE_FIG.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
T6_RATE_FIG.savefig(
    Path("outputs/figures") / "task6_bucket_rates.png",
    dpi=150, bbox_inches="tight",
)

T6_RATE_FIG

T6_VAD_COLUMNS = [
    "valence_mean",
    "arousal_mean",
    "dominance_mean",
]

# Use all annotators, not just the five selected examples.
T6_VAD_DATA = T6_TASK3.join(T6_ACTUAL, how="inner")

T6_VAD_BY_TRAIT = (
    T6_VAD_DATA.groupby("actual_trait")[T6_VAD_COLUMNS]
    .mean()
    .reindex(T6_TRAITS)
)

# Show the number of annotators in each trait group.
T6_VAD_COUNTS = (
    T6_VAD_DATA["actual_trait"]
    .value_counts()
    .reindex(T6_TRAITS, fill_value=0)
)

T6_VAD_FIG, T6_VAD_AX = plt.subplots(figsize=(10, 5))

T6_VAD_BY_TRAIT.rename(columns={
    "valence_mean": "Valence",
    "arousal_mean": "Arousal",
    "dominance_mean": "Dominance",
}).plot(kind="bar", ax=T6_VAD_AX)

T6_VAD_AX.set_title("Mean VAD scores by actual dominant trait")
T6_VAD_AX.set_xlabel("Actual dominant trait")
T6_VAD_AX.set_ylabel("Mean lexicon score")
T6_VAD_AX.set_xticklabels(
    [
        f"{trait}\n(n={T6_VAD_COUNTS[trait]})"
        for trait in T6_TRAITS
    ],
    rotation=25,
    ha="right",
)
T6_VAD_AX.legend(title="VAD dimension")
T6_VAD_FIG.tight_layout()

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

T6_VAD_FIG.savefig(
    Path("outputs/figures") / "task6_vad_by_trait.png",
    dpi=150,
    bbox_inches="tight",
)

T6_VAD_FIG

# SETUP:
from personality import env, llm

env.load_env()

T7_CONSENSUS_RAW = data.load_group_consensus()
T7_EVENTS = data.load_events()
T7_PROFILES = data.load_profiles()
T7_ANSWERS = pd.read_json(
    "data/interim/task7/answers.jsonl", lines=True
)
T7_POLAR = pd.read_json(
    "data/interim/task7/polarity.jsonl", lines=True
)

T7_CONSENSUS_COLS = [f"g{n}_consensus_score" for n in range(1, 7)]
T7_EMO_COLS = [f"g{n}_emotion" for n in range(1, 7)]

T7_CONSISTENT = (T7_CONSENSUS_RAW[T7_CONSENSUS_COLS] > 40).all(axis=1)
T7_DIVERGING = T7_CONSENSUS_RAW[T7_EMO_COLS].nunique(axis=1) > 1

T7_SDS = T7_CONSENSUS_RAW[T7_CONSISTENT & T7_DIVERGING]

pd.DataFrame([
    ("all events", len(T7_CONSENSUS_RAW)),
    ("consistent (criterion 1)", int(T7_CONSISTENT.sum())),
    ("consistent + diverging (both criteria)", len(T7_SDS)),
    ("the paper's published subset size", 413),
], columns=["measure", "count"])

T7_MEMBERS = {
    f"g{n}": [m.strip() for m in str(
        T7_CONSENSUS_RAW.iloc[0][f"g{n}_members"]).split("/")]
    for n in range(1, 7)
}

T7_PERSONAS = llm.cluster_personas(
    T7_PROFILES.set_index("ID"), T7_MEMBERS
)

T7_PERSONAS

# a real, short event text (event id 1180, news domain)
T7_EXAMPLE_EVENT = str(
    T7_EVENTS.set_index("id").loc[1180, "english_item"]
)

T7_EXAMPLES = pd.DataFrame([
    ("general (no personality)", llm.general_prompt(T7_EXAMPLE_EVENT)),
    ("persona (the group's average person)",
     llm.persona_prompt(T7_EXAMPLE_EVENT, T7_PERSONAS["g1"])),
    ("persona + step-by-step",
     llm.persona_prompt_cot(T7_EXAMPLE_EVENT, T7_PERSONAS["g1"])),
], columns=["strategy", "prompt"])

T7_EXAMPLES

T7_OK = T7_ANSWERS[T7_ANSWERS["status"] == "ok"].copy()
# the stored key = "event_id|cluster|strategy"
T7_OK[["event_id", "cluster", "strategy"]] = (
    T7_OK["key"].str.split("|", expand=True)
)
T7_OK["event_id"] = T7_OK["event_id"].astype(int)

T7_LOOKUP = T7_CONSENSUS_RAW.set_index("id")
T7_OK["truth"] = [
    str(T7_LOOKUP.loc[e, f"{c}_emotion"])
    for e, c in zip(T7_OK["event_id"], T7_OK["cluster"])
]
T7_OK["source"] = [
    str(T7_LOOKUP.loc[e, "source"]) for e in T7_OK["event_id"]
]

# our extra test (one change at a time; no persona) ran once per
# event: broadcast each answer to the six groups' ground truths
T7_POLAR_ROWS = []
for _, _row in T7_POLAR[T7_POLAR["status"] == "ok"].iterrows():
    _event = int(_row["key"].split("|")[0])
    for _cluster in [f"g{n}" for n in range(1, 7)]:
        T7_POLAR_ROWS.append({
            "event_id": _event,
            "cluster": _cluster,
            "strategy": "polarity",
            "answer": _row["answer"],
            "truth": str(T7_LOOKUP.loc[_event, f"{_cluster}_emotion"]),
            "source": str(T7_LOOKUP.loc[_event, "source"]),
        })
T7_OK = pd.concat(
    [T7_OK, pd.DataFrame(T7_POLAR_ROWS)], ignore_index=True
)

T7_ROWS = []
for _strategy, _part in T7_OK.groupby("strategy"):
    T7_ROWS.append({
        "strategy": _strategy,
        "judgments": len(_part),
        "top1": llm.accuracy_top1(_part["answer"].tolist(),
                                 _part["truth"].tolist()),
        "top2": llm.accuracy_top2(_part["answer"].tolist(),
                                 _part["truth"].tolist()),
    })
T7_METRICS = pd.DataFrame(T7_ROWS)

T7_MAJORITY = T7_OK["truth"].value_counts(normalize=True).max()
T7_BASELINES = pd.DataFrame([
    ("uniform (1 in 7)", 1 / 7),
    ("most frequent answer", T7_MAJORITY),
], columns=["strategy", "top1"])

# which sources count as news or life; the rest = social
NEWS = {"ABC News", "BBC News", "Independent News",
        "Today's Headlines", "The Paper", "Weibo Trending Topics",
        "WeChat article"}
LIFE = {"FMylife", "BenignExistence", "IUTB"}

T7_DOMAIN_DATA = T7_OK.assign(domain=[
    "news" if s in NEWS else ("life" if s in LIFE else "social")
    for s in T7_OK["source"]
])

T7_DOMAIN_ROWS = []
for (_strategy, _domain), _part in T7_DOMAIN_DATA.groupby(
    ["strategy", "domain"]
):
    T7_DOMAIN_ROWS.append({
        "strategy": _strategy,
        "domain": _domain,
        "judgments": len(_part),
        "top1": _part["answer"].apply(
            lambda preds: preds[0]["Emotion"].lower()
        ).eq(_part["truth"].str.lower()).mean(),
    })
T7_BY_DOMAIN = pd.DataFrame(T7_DOMAIN_ROWS)

T7_BY_DOMAIN


T7_ACC_FIG, T7_ACC_AX = plt.subplots(figsize=(9, 5))
T7_ACC_AX.bar(
    T7_METRICS["strategy"] + " top1", T7_METRICS["top1"],
    label="top-1", alpha=0.8,
)
T7_ACC_AX.bar(
    T7_METRICS["strategy"] + " top2", T7_METRICS["top2"],
    label="top-2", alpha=0.5,
)
for _name, _value in zip(T7_BASELINES["strategy"], T7_BASELINES["top1"]):
    T7_ACC_AX.axhline(_value, linestyle="--", linewidth=1, alpha=0.6)
T7_ACC_AX.set_ylabel("accuracy")
T7_ACC_AX.set_title("The authors' task with our model")
T7_ACC_AX.tick_params(axis="x", rotation=20)
T7_ACC_AX.legend()
T7_ACC_FIG.tight_layout()

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
T7_METRICS.to_csv("outputs/tables/task7_metrics.csv", index=False)
T7_BASELINES.to_csv("outputs/tables/task7_baselines.csv", index=False)
T7_BY_DOMAIN.to_csv("outputs/tables/task7_by_domain.csv", index=False)
T7_ACC_FIG.savefig("outputs/figures/task7_accuracy.png", dpi=150)

T7_ACC_FIG
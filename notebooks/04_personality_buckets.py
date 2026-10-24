import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # 04 - Personality buckets (Task 4)

    One bucket per speaker's strongest trait ("dominant"). The same
    counting as Task 1, grouped by trait instead of person.
    """)
    return


@app.cell
def _():
    # SETUP:
    import matplotlib.pyplot as plt
    import pandas as pd

    from personality import buckets, counting, data

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()
    return PROFILES, SPEAKERS, buckets, counting, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## Who is in which bucket

    Dominant trait = the highest of their five profile scores.
    """)
    return


@app.cell
def _(PROFILES, buckets, pd):
    T4_DOMINANT = buckets.dominant_traits(PROFILES)

    T4_DOMINANT_COUNTS = (
        pd.Series(T4_DOMINANT)
        .value_counts()
        .reindex(buckets.TRAIT_NAMES, fill_value=0)
    )

    T4_DOMINANT_COUNTS
    return


@app.cell
def _(mo):
    mo.md("""
    ## Counting each bucket

    The same word counts as Task 1, over the bucket's texts. Raw totals
    depend on the bucket size. Compare per 1,000 tokens (the table at
    the end).
    """)
    return


@app.cell
def _(PROFILES, SPEAKERS, buckets, counting, pd):
    T4_BUCKETS = buckets.bucket_speakers(SPEAKERS, PROFILES)

    T4_STATS = pd.DataFrame({
        trait: counting.speaker_stats(texts)
        for trait, texts in T4_BUCKETS.items()
    }).T.drop(columns="vocabulary_set").astype(int)

    T4_STATS
    return T4_BUCKETS, T4_STATS


@app.cell
def _(T4_BUCKETS, buckets, counting, plt):
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
    return (T4_FIG,)


@app.cell
def _(T4_FIG, T4_STATS):
    T4_STATS.to_csv("outputs/tables/task4_bucket_stats.csv")

    T4_FIG.savefig(
        "outputs/figures/task4_bucket_evolution.png",
        dpi=150,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-1,000-token rates
    """)
    return


@app.cell
def _(T4_STATS):
    T4_NORMALIZED = T4_STATS.div(T4_STATS["tokens"], axis=0) * 1000
    T4_NORMALIZED[["repetitions", "yes_words", "no_words"]]
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ### Task 4 comment

    **What this says.** The five buckets = nearly identical word rates:
    326-333 repetitions; 1.42-1.48 yes-words; 13.68-14.21 no-words (per
    1,000 tokens). Grouping by trait changes little.

    **Weak spots.** The buckets = very uneven (19 people in
    conscientiousness, 2 in extraversion). The smallest read as
    anecdotes. The texts = the events' words, not the speakers' own.
    """)
    return


if __name__ == "__main__":
    app.run()

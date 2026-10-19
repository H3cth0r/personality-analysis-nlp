import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    from personality import buckets, counting, data

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()
    return PROFILES, SPEAKERS, buckets, counting, pd, plt


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

    for ax, trait in zip(T4_AXES, buckets.TRAIT_NAMES):
        series = counting.chunked_series(T4_BUCKETS[trait], chunks=10)

        ax.plot(series["tokens"], series["repetitions"], label="repetitions")
        ax.plot(series["tokens"], series["no_words"], label="negations")
        ax.plot(series["tokens"], series["yes_words"], label="confirmations")

        ax.set_title(trait)
        ax.tick_params(labelsize=8)

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
def _(T4_STATS):
    T4_NORMALIZED = T4_STATS.div(T4_STATS["tokens"], axis=0) * 1000
    T4_NORMALIZED[["repetitions", "yes_words", "no_words"]]
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 4 comment

    The five dominant-personality buckets show very similar word-count patterns when normalized by tokens. Repetitions range from about 326 to 333 per 1,000 tokens, confirmations from about 1.42 to 1.48, and negations from about 13.68 to 14.21. This suggests that the five buckets do not act very differently in these measures. The large differences in raw totals are mainly explained by the unequal number of speakers in each bucket, especially the large conscientiousness bucket. These results are descriptive only: the buckets contain very different numbers of speakers, and the texts are event reactions rather than each speaker's own writing.
    """)
    return


if __name__ == "__main__":
    app.run()

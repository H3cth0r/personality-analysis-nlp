import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 01 — Speaker stats (Task 1)

    Rule A piles (non-neutral reactions). Text column: `english_item`.
    Owner: A.
    """)
    return


@app.cell
def _():
    # SETUP:
    import pandas as pd
    import matplotlib.pyplot as plt
    from personality import counting, data

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()
    return PROFILES, SPEAKERS, counting, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## The five counts, per speaker
    """)
    return


@app.cell
def _(PROFILES, SPEAKERS, counting, pd):
    ROWS = {
        name: counting.speaker_stats(pile["english_item"])
        for name, pile in SPEAKERS.groupby("annotator")
    }
    STATS = pd.DataFrame(ROWS).T.drop(columns="vocabulary_set").astype(int)
    TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
    TRAIT_NAMES = ["openness", "conscientiousness", "extraversion",
                   "agreeableness", "neuroticism"]
    TRAITS = (PROFILES.set_index("ID")[TRAIT_COLS]
              .rename(columns=dict(zip(TRAIT_COLS, TRAIT_NAMES))))
    JOINED = STATS.join(TRAITS)
    STATS.to_csv("outputs/tables/task1_speaker_stats.csv")
    JOINED
    return JOINED, TRAIT_NAMES


@app.cell
def _(mo):
    mo.md("""
    Vocab size: 7,465–9,710 lemmas. Tokens: 104k–167k.
    Pile size varies per speaker — Rule A effect (see comment below).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Evolution: repetitions, yes-words, no-words vs tokens

    One panel per speaker. Cumulative counts at 10 chunk boundaries
    (reading order, text-granular).
    """)
    return


@app.cell
def _(SPEAKERS, counting, plt):
    from matplotlib.ticker import MaxNLocator

    FIG, AXES = plt.subplots(6, 6, figsize=(17, 14), sharex=False)
    for ax, (name, pile) in zip(AXES.flat, SPEAKERS.groupby("annotator")):
        series = counting.chunked_series(pile["english_item"], chunks=10)
        ax.plot(series["tokens"], series["repetitions"], label="repetitions")
        ax.plot(series["tokens"], series["no_words"], label="negations")
        ax.plot(series["tokens"], series["yes_words"], label="confirmations")
        ax.set_title(f"{name} ({int(pile.shape[0])} texts)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    for ax in AXES.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    AXES[0, 0].legend(fontsize=8)
    FIG.supxlabel("tokens so far")
    FIG.tight_layout()
    FIG.savefig("outputs/figures/task1_evolution.png", dpi=150)
    FIG
    return


@app.cell
def _(mo):
    mo.md("""
    All three counts grow near-linearly with tokens: the word habits are
    steady across each pile, not concentrated in later texts.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Personality (annotator BFI, 0–1)
    """)
    return


@app.cell
def _(JOINED, TRAIT_NAMES):
    CORR = (JOINED[["tokens", "vocabulary_size", "repetitions", "yes_words",
                    "no_words"] + TRAIT_NAMES]
            .corr(method="spearman")
            .loc[TRAIT_NAMES, ["tokens", "vocabulary_size", "repetitions",
                               "yes_words", "no_words"]])
    CORR.round(3).to_csv("outputs/tables/task1_trait_correlations.csv")
    CORR.round(2)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Comment (Task 1)

    - Piles: 1,863–3,018 texts, 104k–167k tokens per speaker. Enough text
      for all counts.
    - Vocab size differs little between speakers (7.5k–9.7k lemmas,
      mean 8,707): everyone reads similar-sized piles of similar content.
    - Counts are high but similar across speakers too (mean 45.7k
      repetitions, 1,928 negations, 199 confirmations per pile).
    - Negations outnumber confirmations ~10:1 in every pile.
    - Correlations are weak (n = 36, max |r| = 0.22, Spearman):
      neuroticism + on all five counts; extraversion and agreeableness −;
      openness and conscientiousness ≈ 0.
    - **Main confound:** Rule A makes pile size trait-dependent —
      reactive (higher-N) people qualify more events. Count-level
      differences therefore partly measure engagement volume, not word
      style. Compare with care.
    - Similarity: speakers vary more in *how much* they engage than in
      *how* they word things — vocabulary sizes cluster tightly.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Outputs

    - `outputs/tables/task1_speaker_stats.csv` — the five counts per speaker
    - `outputs/tables/task1_trait_correlations.csv` — Spearman table
    - `outputs/figures/task1_evolution.png` — the 36-panel evolution plot
    """)
    return


if __name__ == "__main__":
    app.run()

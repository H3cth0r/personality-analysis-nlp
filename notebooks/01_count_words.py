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

    One row = one speaker (E1–E36). Counts span all their texts.

    - **vocabulary_size** — how many *different* words they used
      (lemmatized: "agreed" counts as "agree").
    - **tokens** — how many words in total.
    - **repetitions** — how many times a word is *reused inside one text*.
      A text with 30 words and 25 different ones repeats 5.
    - **yes_words** — agree-words (yes, OK, sure…), counted by our frozen
      list.
    - **no_words** — negation words (no, not, never…), same list idea.

    The right-side columns are their personality scores (0–1), from the
    dataset. Read the traits together with the counts below (Personality
    table).
    """)
    return


@app.cell
def _(PROFILES, SPEAKERS, counting, pd):
    T1_ROWS = {
        name: counting.speaker_stats(pile["english_item"])
        for name, pile in SPEAKERS.groupby("annotator")
    }
    T1_STATS = pd.DataFrame(T1_ROWS).T.drop(columns="vocabulary_set").astype(int)
    T1_TRAIT_COLS = ["Open.", "Cons.", "Extra.", "Agree.", "Neuro."]
    T1_TRAIT_NAMES = ["openness", "conscientiousness", "extraversion",
                   "agreeableness", "neuroticism"]
    T1_TRAITS = (PROFILES.set_index("ID")[T1_TRAIT_COLS]
              .rename(columns=dict(zip(T1_TRAIT_COLS, T1_TRAIT_NAMES))))
    T1_JOINED = T1_STATS.join(T1_TRAITS)
    T1_STATS.to_csv("outputs/tables/task1_speaker_stats.csv")
    T1_JOINED
    return T1_JOINED, T1_TRAIT_NAMES


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

    Each small panel = one speaker. The lines are "how many by now?"
    counters: we walk through their texts in reading order, and at 10
    checkpoints we note the total reached so far.

    - x axis = words read so far.
    - y axis = repetitions / negations / confirmations accumulated so far.

    How to read it:

    - A straight rising line = the habit is *constant* — the person
      repeats (or negates) at the same rate from the first text to the
      last.
    - A curve or jump = the habit changes somewhere along the pile.
    - Compare speakers by **slope**, not by height: a bigger pile just
      makes a longer line.

    What the task wants to see: do the habits *change* as more words are
    used? Below: they barely do — all lines look straight.
    """)
    return


@app.cell
def _(SPEAKERS, counting, plt):
    from matplotlib.ticker import MaxNLocator

    T1_FIG, T1_AXES = plt.subplots(6, 6, figsize=(17, 14), sharex=False)
    for ax, (name, pile) in zip(T1_AXES.flat, SPEAKERS.groupby("annotator")):
        series = counting.chunked_series(pile["english_item"], chunks=10)
        ax.plot(series["tokens"], series["repetitions"], label="repetitions")
        ax.plot(series["tokens"], series["no_words"], label="negations")
        ax.plot(series["tokens"], series["yes_words"], label="confirmations")
        ax.set_title(f"{name} ({int(pile.shape[0])} texts)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    for ax in T1_AXES.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    T1_AXES[0, 0].legend(fontsize=8)
    T1_FIG.supxlabel("tokens so far")
    T1_FIG.tight_layout()
    T1_FIG.savefig("outputs/figures/task1_evolution.png", dpi=150)
    T1_FIG
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

    A **correlation table**. The question it answers: *when a personality
    score is high, does a word count also tend to be high?*

    - rows = the 5 personality traits.
    - columns = the 5 word counts.
    - each cell = a number from **−1 to +1**:
      +1 = the two rise together; 0 = no link; −1 = one rises while the
      other falls.
    - method = Spearman (rank-based; robust to our small count ranges).

    Example reading: neuroticism × yes_words = 0.21 — slightly emotional
    speakers tend to use slightly more agree-words. Small number = weak
    link, a trend across 36 people, never a rule.

    This is the Task-1 question "are some attributes associated with some
    personality patterns?" — the answer sits in the comment below.
    """)
    return


@app.cell
def _(T1_JOINED, T1_TRAIT_NAMES):
    T1_CORR = (T1_JOINED[["tokens", "vocabulary_size", "repetitions", "yes_words",
                    "no_words"] + T1_TRAIT_NAMES]
            .corr(method="spearman")
            .loc[T1_TRAIT_NAMES, ["tokens", "vocabulary_size", "repetitions",
                               "yes_words", "no_words"]])
    T1_CORR.round(3).to_csv("outputs/tables/task1_trait_correlations.csv")
    T1_CORR.round(2)
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

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
    **What the table says:** speakers differ most in their *pile*, not in
    their words.

    - Vocabulary size sits in a tight band: 7.5k–9.7k lemmas (~15% spread)
      across all 36 people.
    - Tokens: 104k–167k (~30% spread) — this follows from pile size,
      i.e. *how many stories made them feel something*.
    - Repetitions: ~45.7k of ~139k tokens — about a third of every pile is
      words said again inside the same text (mostly small common words).
    - Confirmations are rare: ~199 yes-words vs ~1,928 no-words per pile,
      a ~10:1 gap in every speaker.

    So the word habits themselves are remarkably alike. Any personality
    signal must live in the small variations — the correlation table does
    that check below.
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
    **What the grid says:** word habits do not change over the pile.

    - Every line is near-straight: repetitions, negations and confirmations
      accumulate at a constant rate from the first text to the last.
    - The small bends come from chunk boundaries landing between texts
      (texts are not equal-length), not from behavior changes.
    - Meaning: nobody "warms up" or "cools down" mid-pile — so differences
      between people, if any, are *level* differences (slope), not changes
      along the way.
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
    **What the correlation table says:** the only visible ties are one-
    directional and weak.

    - Neuroticism: + with all five counts (0.14–0.21). Reading: more
      emotional people qualify *more* events (bigger piles — Rule A effect)
      and use slightly more yes-words and no-words per event.
    - Extraversion and agreeableness: small negatives (−0.14 to −0.22).
    - Openness and conscientiousness: ≈ 0 — no link with these counts.

    Magnitude check: max |r| = 0.22 at n = 36 is within chance range —
    treat every number as a direction, not an effect.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task 1 verdict

    Word habits are strikingly similar across the 36 speakers; the real
    differentiator is *how much they engage* (pile size), which itself is
    partly personality-shaped (neuroticism). Weak support, then, for
    "word habits show personality" — and partly confounded: under Rule A,
    pile size is trait-dependent by construction. The texts are also the
    events' words, not the speakers' own writing (structural weakness,
    revisited in Task 8). n = 36: nothing here is statistically strong.
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

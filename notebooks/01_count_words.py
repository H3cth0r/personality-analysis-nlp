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
    # 01 - Speaker stats (Task 1)

    Rule A = each person's pile = the events that made them feel
    something. Text column: `english_item`.
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

    One row = one speaker (E1-E36). Counts span all their texts.

    - **vocabulary_size**: how many *different* words they used
      (words counted in their base form: "agreed" counts as "agree").
    - **tokens**: how many words in total.
    - **repetitions**: how many times a word is *reused inside one text*.
      A text with 30 words and 25 different ones repeats 5.
    - **yes_words**: agree-words (yes, OK, sure…), counted by our frozen
      list.
    - **no_words**: negation words (no, not, never…), same list idea.

    Right side = their personality scores (0-1), from the dataset.
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

    - Vocabulary size sits in a tight band: 7.5k-9.7k different words
      (~15% spread)
      across all 36 people.
    - Tokens: 104k-167k (~30% spread). This follows from pile size:
      how many stories made them feel something.
    - Repetitions: ~45.7k of ~139k tokens. About a third of every pile is
      words said again inside the same text (mostly small common words).
    - Confirmations are rare: ~199 yes-words vs ~1,928 no-words per pile,
      a ~10:1 gap in every speaker.

    Word habits barely differ. Any personality signal lives in the small
    variations. See the tie table.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Evolution: repetitions, yes-words, no-words vs tokens

    Each small panel = one speaker. The lines are "how many by now?"
    counters: we walk through their texts in reading order, and at 10
    stops along the way we note the total reached so far.

    - x axis = words read so far.
    - y axis = repetitions / negations / confirmations accumulated so far.

    How to read it:

    - A straight rising line = the habit is *constant*.
    - A curve or jump = the habit changed somewhere along the pile.
    - Compare speakers by **slope**, not by height: a bigger pile just
      makes a longer line.

    What the task wants to see: do the habits *change* as more words are
    used? Below: they barely do. All lines look straight.
    """)
    return


@app.cell
def _(SPEAKERS, counting, plt):
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
    for _ax2 in T1_AXES.flat:
        _ax2.set_xlabel("")
        _ax2.set_ylabel("")
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
    - Meaning: nobody "warms up" or "cools down" mid-pile. Differences
      between people, if any, are *level* differences (slope), not
      changes along the way.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Personality (annotator BFI, 0-1)

    A **correlation table**. The question: *when a personality score is
    high, does a word count also tend to be high?* Cells go from -1
    to +1: +1 = the two rise together, 0 = no link, -1 = one rises
    while the other falls. Method = Spearman (compares rank order,
    not raw size).

    Example reading: neuroticism × yes_words = 0.21, so slightly
    emotional speakers use slightly more agree-words. Weak.
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

    - Neuroticism: + with all five counts (0.14-0.21). Reading: more
      emotional people qualify *more* events (bigger piles, Rule A
      effect) and use slightly more yes-words and no-words per event.
    - Extraversion and agreeableness: small negatives (−0.14 to −0.22).
    - Openness and conscientiousness: ≈ 0. No link with these counts.

    Magnitude check: max |r| = 0.22 at n = 36 is within chance range.
    Treat every number as a direction, not an effect.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task 1 verdict

    Word habits are nearly the same across the 36 speakers. Pile size is
    the real differentiator, and it partly tracks neuroticism (a Rule A
    side effect). Weak support, then, for "word habits show personality".
    The texts are also the events' words, not the speakers' own. n = 36.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Outputs

    - `outputs/tables/task1_speaker_stats.csv`: the five counts per speaker
    - `outputs/tables/task1_trait_correlations.csv`: Spearman table
    - `outputs/figures/task1_evolution.png`: the 36-panel evolution plot
    """)
    return


if __name__ == "__main__":
    app.run()

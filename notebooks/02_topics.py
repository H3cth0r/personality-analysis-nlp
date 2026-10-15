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
    # 02 — Topic change (Task 2)

    Hypothesis: people who change topics introduce new words fast and
    have more topic piles.
    """)
    return


@app.cell
def _():
    # SETUP:
    import pandas as pd
    import matplotlib.pyplot as plt
    from personality import counting, data, topics

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()
    return PROFILES, SPEAKERS, counting, pd, plt, topics


@app.cell
def _(mo):
    mo.md("""
    ## (a) Vocabulary growth vs tokens

    "How many new words by now?" — each curve = one speaker, measured at 10
    checkpoints of their pile.

    - **Left panel:** raw counts. Every speaker climbs from 0 to their
      vocabulary size; curves differ mainly in length (pile size).
    - **Right panel:** same curves, but x = *fraction* of the pile read.
      Now shape is what differs: a curve that keeps rising at the end =
      keeps finding new words late = the topic-changer pattern.
    """)
    return


@app.cell
def _(SPEAKERS, counting, plt):
    T2_FIG, (T2_AX1, T2_AX2) = plt.subplots(1, 2, figsize=(13, 5))
    for _, pile_b in SPEAKERS.groupby("annotator"):
        growth = counting.vocab_growth(pile_b["english_item"])
        T2_AX1.plot(growth["tokens"], growth["vocab_size"], linewidth=1)
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
    return


@app.cell
def _(mo):
    mo.md("""
    Read: in both panels the curves bend flat early — most new words are
    discovered in the first ~20% of the pile, then the discovery rate
    falls. The shapes are similar across speakers; how fast each curve
    flattens is the (weak) difference. Vocabulary growth alone is a blunt
    tool for separating personalities.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## (b) The topic robot (LDA): how many topic piles per speaker?

    For every speaker we train one LDA per candidate topic count
    (k = 3, 5, 8, 12, 16, 20) and score it twice:

    - **coherence (c_v)**, blue, left axis — "do the piles make sense
      together?" Higher = better.
    - **perplexity**, gray dashed, right axis — how sure the model is;
      higher (closer to 0) = better.

    Where coherence peaks = that speaker's best topic count. One panel per
    speaker; results cached in `data/interim/lda/`.
    """)
    return


@app.cell
def _(SPEAKERS, plt, topics):
    T2_SWEEPS = {}
    T2_FIG2, T2_AXES = plt.subplots(6, 6, figsize=(17, 14))
    for ax_b, (name_b, pile_c) in zip(T2_AXES.flat, SPEAKERS.groupby("annotator")):
        result = topics.sweep_speaker(name_b, pile_c["english_item"])
        T2_SWEEPS[name_b] = result
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
    return (T2_SWEEPS,)


@app.cell
def _(PROFILES, T2_SWEEPS, pd, topics):
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
    return (T2_CORR,)


@app.cell
def _(mo):
    mo.md("""
    ## Trait link

    The table above lists each speaker's best topic count. Full tables:
    `outputs/tables/task2_best_topics.csv` (with trait columns) and
    `outputs/tables/task2_trait_correlations.csv` (Spearman).
    """)
    return


@app.cell
def _(T2_CORR):
    T2_CORR
    return


@app.cell
def _(mo):
    mo.md("""
    ## Comment (Task 2)

    **(a) Vocabulary growth — a blunt tool.** All 36 curves share the same
    shape: fast new-word discovery in the first ~20% of the pile, then
    saturation (7.5k–9.7k lemmas at the end, tightly banded). Shapes barely
    differ between speakers — expected: everyone reacts to texts drawn
    from the same event pool, so their vocabulary *sources* overlap.
    Verdict: vocab growth alone discriminates personalities poorly.

    **(b) LDA — the two criteria disagree.** Coherence (c_v) rises with k
    on average (0.31 at k=3 → 0.32 at k=20) while perplexity worsens
    (−6.69 → −7.06): more piles look better to coherence, worse to
    perplexity. "Optimal topics" is therefore criterion-dependent, and we
    pick per speaker by coherence.

    **Best topic counts differ between people**
    (2×k=3, 4×k=5, 2×k=8, 6×k=12, 13×k=16, 9×k=20).

    **Trait link — weak, same story as Task 1** (Spearman, n = 36):
    best_k: agreeableness +0.24, conscientiousness +0.20, extraversion
    +0.13, openness +0.08, neuroticism −0.19. Coherence itself: ≈ 0 with
    every trait.

    **Honest caveats.** (1) c_v coherence is noisy at 2-pass LDA on
    ~2,500 short texts — the k-ordering can shuffle with more passes.
    (2) The texts are the events' words again, not the speaker's own:
    LDA describes what they *reacted to*, not how they *talk about it*.
    (3) n = 36: weak links, directions only.
    """)
    return


if __name__ == "__main__":
    app.run()

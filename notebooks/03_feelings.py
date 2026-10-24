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
    # 03 - Feeling words (Task 3)

    Rule A piles (non-neutral reactions). Text column: `english_item`.
    Dictionary: NRC VAD (valence, arousal, dominance).
    """)
    return


@app.cell
def _():
    # SETUP:
    import pandas as pd
    import matplotlib.pyplot as plt
    from personality import data, vad

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()
    VAD = vad.load_vad()
    return PROFILES, SPEAKERS, VAD, pd, plt, vad


@app.cell
def _(mo):
    mo.md("""
    ## VAD scores, per speaker

    One row = one speaker (E1-E36). Scores come from NRC dictionary
    hits in that speaker's pile, not from every token.

    - **n_vad_hits**: how many tokens sat in the dictionary. The
      divider for the averages above.
    - **valence**: pleasant (1) vs unpleasant (0).
    - **arousal**: excited (1) vs calm (0).
    - **dominance**: in control (1) vs overpowered (0).

    For each dimension: the average, how much the scores wiggle, the
    minimum, the maximum,
    all over the hits only. Right side = their personality scores (0-1).
    """)
    return


@app.cell
def _(PROFILES, SPEAKERS, VAD, pd, vad):
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
    return T3_JOINED, T3_STATS, T3_TRAIT_NAMES


@app.cell
def _(mo):
    mo.md("""
    **What the table says:** the feeling-word *level* barely moves;
    the *pile* does.

    - Valence means sit in a 0.597-0.607 band (about 1% spread).
    - Arousal means: 0.437-0.444. Dominance means: 0.520-0.525.
      Same story: a tight band, not 36 different emotional palettes.
    - Hits: 48k-77k. That range tracks pile size (Task 1 tokens), not a
      different dictionary habit.
    - min/max = the scale's ends for everyone: the extremes live in
      the shared pool, so every pile finds them.

    Feeling-word averages are even more alike than the Task 1 counts.
    Any personality signal has to live in the small leftover variation.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mean valence, arousal, dominance by speaker

    Three series, 36 speakers. x = speaker, y = mean score of dictionary
    hits in that pile.

    How to read it: height differences are the differences between
    people. If the three lines are almost flat, the dictionary is
    describing the *events*, not the people.
    """)
    return


@app.cell
def _(T3_STATS, plt):
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
    return


@app.cell
def _(mo):
    mo.md("""
    **What the plot says:** three flat bands.

    - Valence rides near 0.60, dominance near 0.52, arousal near 0.44.
      The gap *between* dimensions is larger than the gap *between*
      speakers.
    - The wiggles are real but tiny: a few thousandths on a 0-1 scale.
    - No speaker is a high-valence or high-arousal outlier. Under Rule A
      everyone is sampling the same English event texts; they only differ
      in *which* events they react to, and that is not enough to move
      the VAD means.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Personality (annotator BFI, 0-1)

    A **correlation table**. Question: *when a trait score is high, is
    a VAD mean also high?*

    - rows = the 5 personality traits.
    - columns = the 3 VAD means.
    - each cell = Spearman ρ from **−1 to +1**.
    """)
    return


@app.cell
def _(T3_JOINED, T3_TRAIT_NAMES):
    T3_MEAN_COLS = ["valence_mean", "arousal_mean", "dominance_mean"]
    T3_CORR = (T3_JOINED[T3_MEAN_COLS + T3_TRAIT_NAMES]
            .corr(method="spearman")
            .loc[T3_TRAIT_NAMES, T3_MEAN_COLS])
    T3_CORR.round(3).to_csv("outputs/tables/task3_trait_correlations.csv")
    T3_CORR.round(2)
    return


@app.cell
def _(mo):
    mo.md("""
    **What the correlation table says:** one modest tie, the rest noise.

    - Neuroticism × arousal = +0.26 = the strongest cell. More emotional
      speakers' piles sit a little higher on arousal. Same direction as
      Task 1, likely the Rule A effect: they qualify more events.
    - Extraversion: small negatives with all three means (−0.14 to
      −0.20). Agreeableness × dominance = −0.15.
    - Openness and conscientiousness: ≈ 0 against valence; small
      negatives against arousal.

    Magnitude check: max |r| = 0.26 at n = 36 is still a weak trend,
    not a discriminator.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task 3 verdict

    NRC VAD separates nobody. The three means sit a few thousandths
    apart on the 0-1 scale; min/max are shared, because the dictionary
    words live in the events. One weak tie: neuroticism-arousal, easy to
    read as "who reacts to more events" rather than "who uses different
    feeling words". The texts = the events' English, not the speakers'
    own. n = 36.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Outputs

    - `outputs/tables/task3_speaker_vad.csv`: VAD mean/std/min/max and hits per speaker
    - `outputs/tables/task3_trait_correlations.csv`: Spearman table
    - `outputs/figures/task3_vad_means.png`: three mean series, 36 speakers
    """)
    return


if __name__ == "__main__":
    app.run()

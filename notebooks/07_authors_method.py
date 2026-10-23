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
    # 07 - The authors' method (Task 7)

    The dataset authors' question, ours to reproduce: **given a person's
    Big Five profile, how would they emotionally react to an event?**

    - three prompt strategies (no-personality observer / persona /
      persona + think-step-by-step),
    - evaluated with their metrics (top-1 / top-2 accuracy) against what
      the personality groups actually answered (from the consensus file,
      offline),
    - their published numbers as context (top-1 ~22-31%): our model is a
      free one, so the fair comparison = our numbers vs our baselines.

    This skeleton runs WITHOUT any model call: eval set + personas +
    prompt examples only.
    """)
    return


@app.cell
def _():
    # SETUP:
    import pandas as pd
    import matplotlib.pyplot as plt
    from personality import data, env, llm

    env.load_env()
    T7_CONSENSUS_RAW = data.load_group_consensus()
    T7_EVENTS = data.load_events()
    T7_PROFILES = data.load_profiles()
    return data, env, llm, plt, T7_CONSENSUS_RAW, T7_EVENTS, T7_PROFILES


@app.cell
def _(mo):
    mo.md("""
    ## The evaluation set (their two-criteria subset)

    - consistent: every personality group answered with high internal
      agreement (consensus score > 40 of 100)

    - diverging: the groups' majority answers are NOT all the same

    Documented difference: the paper's exact divergence strength is not
    fully specified against the released scores; our variant (threshold
    40/100 + at least two distinct labels) = the closest match to their
    published composition (469 events, mostly news - their 413: news
    257 / social 69 / life 87).
    """)
    return


@app.cell
def _(T7_CONSENSUS_RAW, pd):
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
    return (T7_SDS,)


@app.cell
def _(mo):
    mo.md("""
    ## The personas: one per personality group

    Each group's persona = its members' average Big Five profile (the
    centroid), presented as the paper's persona block.
    """)
    return


@app.cell
def _(T7_CONSENSUS_RAW, T7_PROFILES, llm):
    T7_MEMBERS = {
        f"g{n}": [m.strip() for m in str(
            T7_CONSENSUS_RAW.iloc[0][f"g{n}_members"]).split("/")]
        for n in range(1, 7)
    }
    T7_PERSONAS = llm.cluster_personas(
        T7_PROFILES.set_index("ID"), T7_MEMBERS
    )
    T7_PERSONAS
    return (T7_PERSONAS,)


@app.cell
def _(mo):
    mo.md("""
    ## One worked example per strategy
    """)
    return


@app.cell
def _(T7_EVENTS, T7_PERSONAS, llm):
    # a real, short event text (event id 1180, news domain)
    T7_EXAMPLE_EVENT = str(
        T7_EVENTS.set_index("id").loc[1180, "english_item"]
    )

    T7_EXAMPLES = pd.DataFrame([
        ("general (no personality)", llm.general_prompt(T7_EXAMPLE_EVENT)),
        ("persona (cluster centroid)",
         llm.persona_prompt(T7_EXAMPLE_EVENT, T7_PERSONAS["g1"])),
        ("persona + step-by-step",
         llm.persona_prompt_cot(T7_EXAMPLE_EVENT, T7_PERSONAS["g1"])),
    ], columns=["strategy", "prompt"])
    T7_EXAMPLES
    return (T7_EXAMPLES,)


if __name__ == "__main__":
    app.run()
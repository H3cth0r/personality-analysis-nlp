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
    T7_ANSWERS = pd.read_json(
        "data/interim/task7/answers.jsonl", lines=True
    )
    T7_POLAR = pd.read_json(
        "data/interim/task7/polarity.jsonl", lines=True
    )
    return (data, env, llm, pd, plt, T7_ANSWERS, T7_CONSENSUS_RAW,
            T7_EVENTS, T7_POLAR, T7_PROFILES)


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
def _(T7_EVENTS, T7_PERSONAS, llm, pd):
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


@app.cell
def _(mo):
    mo.md("""
    ## Results: the three strategies vs the group answers

    Answers come from the batch runner (`scripts/run_task7.py`, 100
    seeded events x 6 clusters x 3 strategies); this notebook reads the
    cache. Accuracy compares each first/second guess with the group's
    stored majority emotion. Baselines: uniform (1 in 7) and the most
    frequent answer in the sample.
    """)
    return


@app.cell
def _(T7_ANSWERS, T7_CONSENSUS_RAW, T7_POLAR, llm, pd):
    T7_OK = T7_ANSWERS[T7_ANSWERS["status"] == "ok"].copy()
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

    # the polarity ablation ran once per event (no persona): broadcast
    # each answer to the six clusters' ground truths
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
    return T7_BASELINES, T7_METRICS, T7_OK


@app.cell
def _(T7_BASELINES, T7_METRICS, T7_OK, pd):
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
    return (T7_BY_DOMAIN,)


@app.cell
def _(T7_BY_DOMAIN, T7_METRICS, T7_BASELINES, plt):
    from pathlib import Path

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
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task 7 comment

    **Setup.** Their task, our data: 100 events x 6 personality groups
    x 3 of their prompts, plus one extra test of ours. Model: a cheap
    DeepSeek (not theirs). Truth: each group's most common feeling.

    **Scores (top-1 / top-2, ~600 tests each):**

    - general (no personality): **54.4% / 77.5%**
    - positive/negative first, no persona: 52.7% / 73.2%
    - persona + think step by step: 51.2% / 73.9%
    - persona: 49.7% / 74.0%
    - baselines: uniform 14.3%, commonest answer 20.0%

    **What this says.**

    - All four beat the baselines hard. The model knows the obvious
      feeling of an easy event.
    - Personality makes it WORSE. Plain prompt = best. Two reasons:
      our persona = an averaged person (faint signal), and our target =
      the group's most common answer (a persona can't move a group vote).
    - Our extra test (say positive/negative first, no persona) scores
      like plain (52.7 vs 54.4) and beats persona. So the
      positive/negative step = harmless; the persona block = what costs
      points. Improvement attempt = a NEGATIVE result. Honest.
    - News = hardest (46-49%). Social and life = easier (55-62%). The
      paper says social is hardest - our sample differs, so no match
      there.

    **The paper comparison, honest.** Our ~50% is NOT their 22-31%.
    Different tests: their sample = events where groups really DISAGREE
    (hard). Ours = mostly events where groups AGREE (easy). Ours asks
    "can you spot the obvious feeling?" Theirs asks "can you pick the
    group's side when sides differ?" On easy events we beat baselines;
    on hard events even their big models stay low.

    **Caveats.** Texts = the events' words, not the writers' own; 100
    events per prompt; our model is not theirs; our subset is our
    rebuilt version (documented above); 1,894 of 1,900 calls worked
    (network/parse misses, retried).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Outputs

    - `outputs/tables/task7_metrics.csv` — accuracy per strategy
    - `outputs/tables/task7_by_domain.csv` — the same, split by domain
    - `outputs/figures/task7_accuracy.png` — the comparison chart
    - answers cache: `data/interim/task7/answers.jsonl` (gitignored)
    """)
    return


if __name__ == "__main__":
    app.run()
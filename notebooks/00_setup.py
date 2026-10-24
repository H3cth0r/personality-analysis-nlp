import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 00 - Setup

        Load Persona-E2. Run checks. Build `events.parquet`.
        """
    )
    return


@app.cell
def _():
    # SETUP:
    from personality import data

    RAW = data.load_raw()
    PROFILES = data.load_profiles()
    GROUPS = data.load_group_consensus()
    return GROUPS, PROFILES, RAW, data


@app.cell
def _(GROUPS, PROFILES, RAW, data, mo):
    mo.md("## Checks")
    return


@app.cell
def _(GROUPS, PROFILES, RAW, data):
    CHECKS = data.check_events(RAW, PROFILES, groups=GROUPS)
    return (CHECKS,)


@app.cell
def _(CHECKS, mo):
    CHECKS
    return


@app.cell
def _(mo):
    mo.md("## Build `events.parquet`")
    return


@app.cell
def _(RAW, data):
    EVENTS = data.build_events(RAW)
    data.save_events(EVENTS)
    EVENTS.shape
    return (EVENTS,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Notes

        - Raw file has 3,113 events, paper says 3,111. Two extra rows. Keep them.
        - `nan` among emotion labels = skipped reactions by some annotators.
        - `events.parquet` = wide format, one row per event.
        """
    )
    return


if __name__ == "__main__":
    app.run()
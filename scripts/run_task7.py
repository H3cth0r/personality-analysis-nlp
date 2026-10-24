"""Task-7 full run: 100 events x 6 clusters x 3 strategies = 1,800 calls.

Answers append to data/interim/task7/answers.jsonl; re-runs skip them.
Usage:  uv run python scripts/run_task7.py
"""
import json
from pathlib import Path

from personality import data, env, llm

MODEL = "deepseek/deepseek-chat-v3-0324"

env.load_env()

consensus = data.load_group_consensus()
profiles = data.load_profiles().set_index("ID")
events = data.load_events().set_index("id")

scores = [f"g{n}_consensus_score" for n in range(1, 7)]
labels = [f"g{n}_emotion" for n in range(1, 7)]
sds = consensus[(consensus[scores] > 40).all(axis=1)
                & (consensus[labels].nunique(axis=1) > 1)]
sample = sds.sample(n=100, random_state=42)

members = {f"g{n}": str(consensus.iloc[0][f"g{n}_members"]).split("/")
           for n in range(1, 7)}
personas = llm.cluster_personas(profiles, members)

path = Path("data/interim/task7/answers.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
done = set()
if path.exists():
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["status"] == "ok":
            done.add(row["key"])

count = errors = 0
for event_id in sample["id"]:
    for cluster in members:
        for strategy in ("general", "persona", "cot"):
            key = f"{event_id}|{cluster}|{strategy}"
            if key in done:
                continue
            text = str(events.loc[event_id, "english_item"])
            if strategy == "general":
                prompt = llm.general_prompt(text)
            elif strategy == "persona":
                prompt = llm.persona_prompt(text, personas[cluster])
            else:
                prompt = llm.persona_prompt_cot(text, personas[cluster])
            try:
                answer = llm.parse_predictions(llm.run_prompt(prompt, model=MODEL))
                row = {"key": key, "model": MODEL, "status": "ok", "answer": answer}
            except RuntimeError as error:
                errors += 1
                row = {"key": key, "model": MODEL, "status": "error",
                       "error": str(error)[:300]}
            except ValueError as error:
                errors += 1
                row = {"key": key, "model": MODEL, "status": "parse_error",
                       "error": str(error)[:300]}
            with open(path, "a", encoding="utf-8") as cache:
                cache.write(json.dumps(row) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"{count} done, {errors} errors so far", flush=True)

print(f"finished: {count} new calls, {errors} errors", flush=True)
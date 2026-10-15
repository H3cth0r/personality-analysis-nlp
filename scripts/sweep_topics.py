"""Run the LDA sweep for every speaker, once.

Results cache to data/interim/lda/<speaker>_topics.csv — re-running
this script only fills what is missing.

Run as:  uv run python scripts/sweep_topics.py
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("fork")  # macOS defaults to spawn; gensim needs fork

    from personality import data, topics

    speakers = data.load_speakers()
    for name, pile in speakers.groupby("annotator"):
        result = topics.sweep_speaker(name, pile["english_item"])
        print(f"{name}: best k = {topics.best_k(result)}", flush=True)
    print("sweep complete")
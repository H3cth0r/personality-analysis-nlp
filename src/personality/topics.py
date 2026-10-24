"""Topic modeling (Task 2). LDA + coherence + perplexity.

The topic-count sweep is the heavy part, so results are cached per
speaker in data/interim/lda/ — only the first run pays for it.
"""
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

from personality.text import tokenize

TOPIC_COUNTS = [3, 5, 8, 12, 16, 20]
INTERIM_DIR = Path("data/interim/lda")


def sweep(texts, ks=TOPIC_COUNTS):
    """Train one LDA per topic count; score coherence (c_v) and perplexity.

    Returns a DataFrame: k, coherence (higher = better), perplexity
    (log likelihood; higher = better).
    """
    texts = [[w for w in tokenize(t) if w] for t in texts]
    texts = [t for t in texts if len(t) >= 10]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(d) for d in texts]
    rows = []
    for k in ks:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                       random_state=42, passes=2)
        coherence = CoherenceModel(model=lda, texts=texts,
                                   dictionary=dictionary,
                                   coherence="c_v").get_coherence()
        rows.append({"k": k, "coherence": coherence,
                     "perplexity": lda.log_perplexity(corpus)})
    return pd.DataFrame(rows)


def sweep_speaker(name, texts, ks=TOPIC_COUNTS):
    """The sweep with a per-speaker cache file under data/interim/lda/."""
    path = INTERIM_DIR / f"{name}_topics.csv"
    if path.exists():
        return pd.read_csv(path)
    result = sweep(texts, ks)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def best_k(sweep_result):
    """The topic count with the highest coherence at each speaker."""
    return int(sweep_result.loc[sweep_result["coherence"].idxmax(), "k"])


def load_all_sweeps():
    """All cached sweeps as one DataFrame with an `annotator` column."""
    frames = []
    for path in sorted(INTERIM_DIR.glob("*_topics.csv")):
        df = pd.read_csv(path)
        df.insert(0, "annotator", path.stem.replace("_topics", ""))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
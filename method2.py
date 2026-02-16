# method2.py
import pandas as pd
import numpy as np
from student_t_mm import StudentTMixtureRobust


def compute_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1)).dropna()


def label_regimes(mu: np.ndarray) -> list:
    """
    Label regimes economically:
    0: BEAR_VOLATILE
    1: BEAR_CALM
    2: BULL_VOLATILE
    3: BULL_CALM
    Sorted by mean return.
    """
    order = np.argsort(mu)
    labels = [None] * len(mu)
    regime_names = ["BEAR_VOLATILE", "BEAR_CALM", "BULL_VOLATILE", "BULL_CALM"]
    for idx, o in enumerate(order):
        labels[o] = regime_names[idx]
    return labels


def regime_confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def apply_method_2(price: pd.Series, n_components: int = 4):
    ret = compute_returns(price)

    model = StudentTMixtureRobust(n_components=n_components)
    model.fit(ret.values.reshape(-1, 1))

    probs = model.predict_proba(ret.values.reshape(-1, 1))
    states = probs.argmax(axis=1)
    conf = regime_confidence(probs)
    labels = label_regimes(model.mu)
    regime_names = [labels[s] for s in states]

    df = pd.DataFrame(index=ret.index)
    df["regime_id"] = states
    df["regime"] = regime_names
    df["confidence"] = conf
    df["return"] = ret
    df["top_prob"] = probs.max(axis=1)

    return df, model, probs

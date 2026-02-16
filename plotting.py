# plotting.py
import matplotlib.pyplot as plt
import numpy as np


def plot_method_2_results(price, regimes, model, probs):
    """
    Simple plot of price with shaded regimes.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(price.index, price.values, color="black", label="SP500 Price")

    colors = {
        "BEAR_VOLATILE": "red",
        "BEAR_CALM": "orange",
        "BULL_VOLATILE": "blue",
        "BULL_CALM": "green",
    }

    last_regime = regimes["regime"].iloc[0]
    start_idx = 0

    for i, r in enumerate(regimes["regime"]):
        if r != last_regime or i == len(regimes) - 1:
            end_idx = i
            plt.axvspan(
                price.index[start_idx],
                price.index[end_idx - 1],
                color=colors[last_regime],
                alpha=0.2,
            )
            start_idx = i
            last_regime = r

    plt.title("S&P 500 Price with Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# run.py
from data_loader import load_sp500
from method2 import apply_method_2
from plotting import plot_method_2_results


def main():
    price = load_sp500(start="2000-01-01")
    print("Fitting 4-state Student's t Mixture Model...")
    regimes, model, probs = apply_method_2(price, n_components=4)

    print("\nRecent Regimes:")
    print(regimes.tail(20))

    print("\nREGIME DISTRIBUTION:")
    regime_counts = regimes["regime"].value_counts()
    for regime, count in regime_counts.items():
        perc = count / len(regimes) * 100
        avg_conf = regimes[regimes["regime"] == regime]["confidence"].mean()
        print(
            f"{regime:20}: {count:5d} days ({perc:5.1f}%), Avg Confidence: {avg_conf:.3f}"
        )

    plot_method_2_results(price.iloc[1:], regimes, model, probs)


if __name__ == "__main__":
    main()

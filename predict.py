from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi buah dari input fitur CSV.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/best_model.joblib"),
        help="Path model hasil training.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV berisi fitur tanpa kolom target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    data = pd.read_csv(args.input)
    predictions = model.predict(data)

    result = data.copy()
    result["prediction"] = predictions
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

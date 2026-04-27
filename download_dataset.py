from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET = "joshmcadams/oranges-vs-grapefruit"
OUTPUT_PATH = Path("data/oranges_vs_grapefruit.csv")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV di dataset: {dataset_dir}")

    shutil.copy2(csv_files[0], OUTPUT_PATH)
    print(f"Dataset disimpan ke {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

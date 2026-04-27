from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


RANDOM_STATE = 42
TARGET_COLUMN_CANDIDATES = ("name", "class", "label", "fruit")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Download dataset Kaggle oranges-vs-grapefruit, lalu simpan file CSV "
            "ke folder data/oranges_vs_grapefruit.csv."
        )

    data = pd.read_csv(path)
    data.columns = [column.strip().lower().replace(" ", "_") for column in data.columns]
    return data


def find_target_column(data: pd.DataFrame) -> str:
    for column in TARGET_COLUMN_CANDIDATES:
        if column in data.columns:
            return column

    last_column = data.columns[-1]
    if data[last_column].nunique() <= 10:
        return last_column

    raise ValueError(
        "Kolom target tidak ditemukan. Pastikan dataset memiliki kolom name/class/label/fruit."
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    test_size: float,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    target_column = find_target_column(data)

    x = data.drop(columns=[target_column])
    y = data[target_column].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)

    models = {
        # Decision Tree Training
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        # Naive Bayes Training
        "Naive Bayes": GaussianNB(),
        # Support Vector Training
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    reports = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        # Training model
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

        results.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )
        reports.append(f"## {model_name}\n\nAccuracy: {accuracy:.4f}\n\n```\n{report}\n```\n")
        trained_pipelines[model_name] = pipeline

        file_stem = model_name.lower().replace(" ", "_")
        joblib.dump(pipeline, output_dir / "models" / f"{file_stem}.joblib")

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=pipeline.classes_,
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / f"confusion_matrix_{file_stem}.png", dpi=150)
        plt.close()

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    joblib.dump(trained_pipelines[best_model_name], output_dir / "best_model.joblib")

    with (output_dir / "classification_reports.md").open("w", encoding="utf-8") as file:
        file.write("# Classification Reports\n\n")
        file.write(f"Dataset: `{data_path}`\n\n")
        file.write(f"Target column: `{target_column}`\n\n")
        file.write(f"Best model: **{best_model_name}**\n\n")
        file.write(results_df.to_markdown(index=False))
        file.write("\n\n")
        file.write("\n".join(reports))

    make_exploration_figures(data, target_column, output_dir)
    make_tree_figure(trained_pipelines["Decision Tree"], output_dir)

    return results_df


def make_exploration_figures(data: pd.DataFrame, target_column: str, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x=target_column, hue=target_column, palette="Set2", legend=False)
    plt.title("Distribusi Kelas Buah")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "class_distribution.png", dpi=150)
    plt.close()

    numeric_data = data.select_dtypes(include=["number"])
    if numeric_data.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Korelasi Fitur Numerik")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "feature_correlation.png", dpi=150)
        plt.close()


def make_tree_figure(decision_tree_pipeline: Pipeline, output_dir: Path) -> None:
    preprocessor = decision_tree_pipeline.named_steps["preprocessor"]
    tree = decision_tree_pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = None

    plt.figure(figsize=(18, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=decision_tree_pipeline.classes_,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Visualisasi Decision Tree")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "decision_tree.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latih dan bandingkan model klasifikasi jeruk vs anggur."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oranges_vs_grapefruit.csv"),
        help="Path file CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Folder penyimpanan hasil training dan evaluasi.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data uji.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_and_evaluate(args.data, args.output, args.test_size)
    print("Perbandingan model:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

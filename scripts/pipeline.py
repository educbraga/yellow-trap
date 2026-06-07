from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "artifacts" / "matplotlib_cache"))

from classifier import evaluate_classifier, prepare_cropped_dataset, train_knn_classifier


def run_all_steps() -> None:
    prepare_cropped_dataset()
    train_knn_classifier()
    evaluate_classifier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline do classificador de insetos em armadilhas amarelas."
    )
    parser.add_argument(
        "step",
        choices=["prepare", "train", "evaluate", "all"],
        help="Etapa que sera executada.",
    )
    args = parser.parse_args()

    if args.step == "prepare":
        prepare_cropped_dataset()
    elif args.step == "train":
        train_knn_classifier()
    elif args.step == "evaluate":
        evaluate_classifier()
    else:
        run_all_steps()


if __name__ == "__main__":
    main()

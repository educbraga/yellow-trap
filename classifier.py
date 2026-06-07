from __future__ import annotations

import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "yellow-sticky-traps-dataset-main"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATA_DIR = BASE_DIR / "data"
CROPPED_IMAGES_DIR = DATA_DIR / "crops"
CROPS_MANIFEST_PATH = DATA_DIR / "crops_manifest.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_REFERENCE_DIR = BASE_DIR / "static" / "reference"
MEDIA_UPLOADS_DIR = BASE_DIR / "media" / "uploads"

os.environ.setdefault("TORCH_HOME", str(ARTIFACTS_DIR / "torch_cache"))

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_STATE = 42
CROPPED_IMAGE_SIZE = 224
MAX_IMAGES_PER_CLASS = 600
GROUP_SPLIT_FOLDS = 5
NEIGHBORS_TO_SHOW = 5

INSECT_CLASSES = {
    "MR": "Macrolophus pygmaeus",
    "NC": "Nesidiocoris tenuis",
    "WF": "Trialeurodes vaporariorum",
}


def artifacts_are_ready() -> bool:
    return (
        (ARTIFACTS_DIR / "knn.joblib").exists()
        and (ARTIFACTS_DIR / "reference_embeddings.npz").exists()
        and (ARTIFACTS_DIR / "metadata.json").exists()
    )


def load_metrics() -> dict:
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def list_demo_images(limit_per_class: int = 3) -> list[dict]:
    demo_images = []
    for label, class_name in INSECT_CLASSES.items():
        class_dir = STATIC_REFERENCE_DIR / label
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.jpg"))[:limit_per_class]:
            demo_images.append(
                {
                    "label": label,
                    "class_name": class_name,
                    "filename": image_path.name,
                    "url": f"/static/reference/{label}/{image_path.name}",
                }
            )
    return demo_images


def validate_demo_image_path(label: str, filename: str) -> Path:
    if label not in INSECT_CLASSES:
        raise FileNotFoundError("Classe de exemplo nao encontrada.")
    image_path = (STATIC_REFERENCE_DIR / label / filename).resolve()
    if STATIC_REFERENCE_DIR.resolve() not in image_path.parents or not image_path.exists():
        raise FileNotFoundError("Imagem de exemplo nao encontrada.")
    return image_path


@lru_cache(maxsize=1)
def load_feature_extractor():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model, weights.transforms()


def extract_image_embedding(image_path: Path | str) -> np.ndarray:
    model, preprocess = load_feature_extractor()
    image = Image.open(image_path).convert("RGB")
    batch = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(batch).squeeze(0).cpu().numpy().astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def extract_many_embeddings(image_paths: list[Path]) -> np.ndarray:
    return np.vstack([extract_image_embedding(path) for path in image_paths])


def load_model_artifacts():
    if not artifacts_are_ready():
        raise FileNotFoundError(
            "Artefatos nao encontrados. Rode: python scripts/pipeline.py all."
        )

    knn = joblib.load(ARTIFACTS_DIR / "knn.joblib")
    reference_embeddings = np.load(ARTIFACTS_DIR / "reference_embeddings.npz", allow_pickle=True)
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return knn, reference_embeddings, metadata


def predict_insect(image_path: Path) -> dict:
    knn, reference_embeddings, metadata = load_model_artifacts()
    embedding = extract_image_embedding(image_path).reshape(1, -1)
    predicted_label = str(knn.predict(embedding)[0])

    probabilities = knn.predict_proba(embedding)[0]
    probability_by_label = dict(zip(knn.classes_, probabilities))
    confidence = float(probability_by_label.get(predicted_label, 0.0))

    distances, indexes = knn.kneighbors(
        embedding,
        n_neighbors=min(NEIGHBORS_TO_SHOW, len(reference_embeddings["labels"])),
    )
    similar_images = []
    for distance, index in zip(distances[0], indexes[0]):
        neighbor_label = str(reference_embeddings["labels"][index])
        similar_images.append(
            {
                "label": neighbor_label,
                "class_name": INSECT_CLASSES.get(neighbor_label, neighbor_label),
                "distance": float(distance),
                "url": str(reference_embeddings["static_urls"][index]),
            }
        )

    return {
        "prediction": {
            "label": predicted_label,
            "class_name": INSECT_CLASSES.get(predicted_label, predicted_label),
            "confidence": confidence,
            "probabilities": [
                {
                    "label": str(label),
                    "class_name": INSECT_CLASSES.get(str(label), str(label)),
                    "probability": float(probability),
                }
                for label, probability in sorted(probability_by_label.items())
            ],
            "classes": metadata.get("classes", INSECT_CLASSES),
        },
        "neighbors": similar_images,
    }


def calculate_box_with_margin(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: float = 0.12,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    box_width = xmax - xmin
    box_height = ymax - ymin
    horizontal_margin = int(box_width * margin)
    vertical_margin = int(box_height * margin)
    return (
        max(0, xmin - horizontal_margin),
        max(0, ymin - vertical_margin),
        min(image_width, xmax + horizontal_margin),
        min(image_height, ymax + vertical_margin),
    )


def collect_annotated_insects() -> dict[str, list[dict]]:
    insects_by_class: dict[str, list[dict]] = defaultdict(list)
    for annotation_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        root = ET.parse(annotation_path).getroot()
        filename = root.findtext("filename") or f"{annotation_path.stem}.jpg"
        image_path = DATASET_IMAGES_DIR / filename
        if not image_path.exists():
            image_path = DATASET_IMAGES_DIR / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            continue

        for object_index, object_node in enumerate(root.findall("object")):
            label = (object_node.findtext("name") or "").strip()
            box_node = object_node.find("bndbox")
            if label not in INSECT_CLASSES or box_node is None:
                continue

            box = tuple(
                int(float(box_node.findtext(key, "0")))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            insects_by_class[label].append(
                {
                    "image_path": image_path,
                    "box": box,
                    "crop_id": f"{annotation_path.stem}_{object_index:04d}",
                    "source_image_id": annotation_path.stem,
                    "annotation_path": annotation_path,
                }
            )
    return insects_by_class


def prepare_cropped_dataset() -> None:
    random.seed(RANDOM_STATE)
    if not ANNOTATIONS_DIR.exists() or not DATASET_IMAGES_DIR.exists():
        raise SystemExit(
            "Dataset nao encontrado. Extraia yellow-sticky-traps-dataset-main.zip na raiz."
        )

    insects_by_class = collect_annotated_insects()
    if CROPPED_IMAGES_DIR.exists():
        shutil.rmtree(CROPPED_IMAGES_DIR)
    if STATIC_REFERENCE_DIR.exists():
        shutil.rmtree(STATIC_REFERENCE_DIR)
    CROPPED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_crops = 0

    for label, insects in sorted(insects_by_class.items()):
        random.shuffle(insects)
        selected_insects = insects[:MAX_IMAGES_PER_CLASS]
        crop_class_dir = CROPPED_IMAGES_DIR / label
        reference_class_dir = STATIC_REFERENCE_DIR / label
        crop_class_dir.mkdir(parents=True, exist_ok=True)
        reference_class_dir.mkdir(parents=True, exist_ok=True)

        for insect in selected_insects:
            with Image.open(insect["image_path"]).convert("RGB") as image:
                original_box = insect["box"]
                margin_box = calculate_box_with_margin(original_box, image.width, image.height)
                crop = image.crop(margin_box).resize(
                    (CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE),
                    Image.Resampling.LANCZOS,
                )
                filename = f"{label}_{insect['crop_id']}.jpg"
                crop_path = crop_class_dir / filename
                reference_path = reference_class_dir / filename
                crop.save(crop_path, quality=92)
                crop.save(reference_path, quality=92)
                manifest_rows.append(
                    {
                        "crop_path": str(crop_path.relative_to(BASE_DIR)),
                        "static_url": f"/static/reference/{label}/{filename}",
                        "label": label,
                        "class_name": INSECT_CLASSES[label],
                        "source_image_id": insect["source_image_id"],
                        "source_image_path": str(insect["image_path"]),
                        "annotation_path": str(insect["annotation_path"]),
                        "bbox_xmin": original_box[0],
                        "bbox_ymin": original_box[1],
                        "bbox_xmax": original_box[2],
                        "bbox_ymax": original_box[3],
                        "margin_bbox_xmin": margin_box[0],
                        "margin_bbox_ymin": margin_box[1],
                        "margin_bbox_xmax": margin_box[2],
                        "margin_bbox_ymax": margin_box[3],
                    }
                )
                total_crops += 1

        print(f"{label}: {len(selected_insects)} recortes ({INSECT_CLASSES[label]})")

    if not manifest_rows:
        raise SystemExit("Nenhum inseto valido foi encontrado nas anotacoes.")

    with CROPS_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Total: {total_crops} recortes salvos em {CROPPED_IMAGES_DIR}")
    print(f"Manifesto salvo em {CROPS_MANIFEST_PATH}")


def load_crops_manifest() -> list[dict]:
    if not CROPS_MANIFEST_PATH.exists():
        raise SystemExit("Manifesto nao encontrado. Rode python scripts/pipeline.py prepare.")
    with CROPS_MANIFEST_PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    valid_rows = [row for row in rows if (BASE_DIR / row["crop_path"]).exists()]
    if not valid_rows:
        raise SystemExit("Nenhum recorte valido encontrado no manifesto.")
    return valid_rows


def count_classes(labels: list[str] | np.ndarray) -> dict[str, int]:
    counts = Counter(str(label) for label in labels)
    return {label: counts.get(label, 0) for label in sorted(INSECT_CLASSES)}


def train_knn_classifier() -> None:
    rows = load_crops_manifest()
    image_paths = np.array([BASE_DIR / row["crop_path"] for row in rows], dtype=object)
    labels = np.array([row["label"] for row in rows])
    source_image_groups = np.array([row["source_image_id"] for row in rows])
    static_urls = np.array([row["static_url"] for row in rows])

    splitter = StratifiedGroupKFold(
        n_splits=GROUP_SPLIT_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indexes, test_indexes = next(splitter.split(image_paths, labels, source_image_groups))

    train_paths = image_paths[train_indexes].tolist()
    test_paths = image_paths[test_indexes].tolist()
    train_labels = labels[train_indexes].tolist()
    test_labels = labels[test_indexes].tolist()
    train_groups = source_image_groups[train_indexes]
    test_groups = source_image_groups[test_indexes]
    train_static_urls = static_urls[train_indexes].tolist()

    group_overlap = sorted(set(train_groups) & set(test_groups))
    if group_overlap:
        raise SystemExit(
            f"Split invalido: {len(group_overlap)} imagens aparecem em treino e teste."
        )

    print(f"Extraindo embeddings de treino: {len(train_paths)} imagens")
    train_embeddings = extract_many_embeddings(train_paths)
    print(f"Extraindo embeddings de teste: {len(test_paths)} imagens")
    test_embeddings = extract_many_embeddings(test_paths)

    knn = KNeighborsClassifier(
        n_neighbors=NEIGHBORS_TO_SHOW,
        metric="cosine",
        algorithm="brute",
        weights="distance",
    )
    knn.fit(train_embeddings, train_labels)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, ARTIFACTS_DIR / "knn.joblib")
    np.savez_compressed(
        ARTIFACTS_DIR / "reference_embeddings.npz",
        embeddings=train_embeddings,
        labels=np.array(train_labels),
        paths=np.array([str(path) for path in train_paths]),
        static_urls=np.array(train_static_urls),
        groups=np.array(train_groups),
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "test_embeddings.npz",
        embeddings=test_embeddings,
        labels=np.array(test_labels),
        paths=np.array([str(path) for path in test_paths]),
        groups=np.array(test_groups),
    )
    metadata = {
        "classes": INSECT_CLASSES,
        "k_neighbors": NEIGHBORS_TO_SHOW,
        "metric": "cosine",
        "split_strategy": "stratified_group_kfold_by_source_image",
        "n_splits": GROUP_SPLIT_FOLDS,
        "random_state": RANDOM_STATE,
        "train_size": len(train_paths),
        "test_size": len(test_paths),
        "train_groups": len(set(train_groups)),
        "test_groups": len(set(test_groups)),
        "group_overlap_count": len(group_overlap),
        "train_class_counts": count_classes(train_labels),
        "test_class_counts": count_classes(test_labels),
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Artefatos salvos em {ARTIFACTS_DIR}")


def evaluate_classifier() -> None:
    knn_path = ARTIFACTS_DIR / "knn.joblib"
    test_embeddings_path = ARTIFACTS_DIR / "test_embeddings.npz"
    reference_embeddings_path = ARTIFACTS_DIR / "reference_embeddings.npz"
    metadata_path = ARTIFACTS_DIR / "metadata.json"
    if not knn_path.exists() or not test_embeddings_path.exists():
        raise SystemExit("Artefatos nao encontrados. Rode python scripts/pipeline.py train.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    knn = joblib.load(knn_path)
    test_data = np.load(test_embeddings_path, allow_pickle=True)
    reference_data = np.load(reference_embeddings_path, allow_pickle=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    train_groups = set(str(group) for group in reference_data["groups"])
    test_groups = set(str(group) for group in test_data["groups"])
    group_overlap_count = len(train_groups & test_groups)
    if group_overlap_count > 0:
        raise SystemExit(
            f"Avaliacao bloqueada: {group_overlap_count} imagens aparecem em treino e teste."
        )

    test_embeddings = test_data["embeddings"]
    expected_labels = test_data["labels"]
    predicted_labels = knn.predict(test_embeddings)
    predicted_probabilities = knn.predict_proba(test_embeddings)
    labels = list(knn.classes_)

    report = classification_report(
        expected_labels,
        predicted_labels,
        labels=labels,
        target_names=[INSECT_CLASSES.get(label, label) for label in labels],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(expected_labels, predicted_labels, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de confusao")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    roc_auc = None
    if len(labels) > 1:
        expected_binary_labels = label_binarize(expected_labels, classes=labels)
        roc_auc = float(
            roc_auc_score(
                expected_binary_labels,
                predicted_probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
        plt.figure(figsize=(7, 6))
        for index, label in enumerate(labels):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                expected_binary_labels[:, index],
                predicted_probabilities[:, index],
            )
            plt.plot(false_positive_rate, true_positive_rate, label=f"{label}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC one-vs-rest (AUC macro={roc_auc:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "roc_auc.png", dpi=160)
        plt.close()

    payload = {
        "classes": INSECT_CLASSES,
        "labels": labels,
        "split": {
            "strategy": metadata.get("split_strategy"),
            "n_splits": metadata.get("n_splits"),
            "random_state": metadata.get("random_state"),
            "train_size": metadata.get("train_size"),
            "test_size": metadata.get("test_size"),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap_count": group_overlap_count,
            "train_class_counts": metadata.get("train_class_counts"),
            "test_class_counts": metadata.get("test_class_counts"),
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "roc_auc_macro_ovr": roc_auc,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Metricas salvas em {REPORTS_DIR}")

from __future__ import annotations

import mimetypes
from pathlib import Path
from uuid import uuid4

from emmett import App, request, response, url

from classifier import (
    ALLOWED_UPLOAD_EXTENSIONS,
    INSECT_CLASSES,
    MEDIA_UPLOADS_DIR,
    REPORTS_DIR,
    artifacts_are_ready,
    list_demo_images,
    load_metrics,
    predict_insect,
    validate_demo_image_path,
)


app = App(__name__)


@app.route("/", template="index.html", methods="get")
async def index():
    return {
        "artifacts_ready": artifacts_are_ready(),
        "classes": INSECT_CLASSES,
        "demo_images": list_demo_images(),
        "error": None,
    }


@app.route("/predict", template="result.html", methods="post")
async def predict():
    files = await request.files
    upload = files.image
    if upload is None:
        return {
            "error": "Envie uma imagem de inseto ja recortada.",
            "prediction": None,
            "neighbors": [],
        }

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        return {
            "error": "Formato invalido. Use JPG, JPEG, PNG ou WEBP.",
            "prediction": None,
            "neighbors": [],
        }

    MEDIA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = MEDIA_UPLOADS_DIR / f"{uuid4().hex}{suffix}"
    await upload.save(str(upload_path))

    try:
        result = predict_insect(upload_path)
    except FileNotFoundError as exc:
        return {
            "error": str(exc),
            "prediction": None,
            "neighbors": [],
        }
    except Exception as exc:
        return {
            "error": f"Falha ao classificar a imagem: {exc}",
            "prediction": None,
            "neighbors": [],
        }

    return {
        "error": None,
        "prediction": result["prediction"],
        "neighbors": result["neighbors"],
        "upload_url": url("media_file", ["uploads", upload_path.name]),
    }


@app.route("/sample/<str:label>/<str:filename>", template="result.html", methods="get")
async def predict_demo_image(label: str, filename: str):
    try:
        image_path = validate_demo_image_path(label, filename)
        result = predict_insect(image_path)
    except FileNotFoundError as exc:
        return {
            "error": str(exc),
            "prediction": None,
            "neighbors": [],
            "upload_url": None,
        }
    except Exception as exc:
        return {
            "error": f"Falha ao classificar a imagem: {exc}",
            "prediction": None,
            "neighbors": [],
            "upload_url": None,
        }

    return {
        "error": None,
        "prediction": result["prediction"],
        "neighbors": result["neighbors"],
        "upload_url": f"/static/reference/{label}/{filename}",
    }


@app.route("/metrics", template="metrics.html", methods="get")
async def metrics():
    return {
        "metrics": load_metrics(),
        "confusion_matrix_url": url("media_file", ["reports", "confusion_matrix.png"])
        if (REPORTS_DIR / "confusion_matrix.png").exists()
        else None,
        "roc_auc_url": url("media_file", ["reports", "roc_auc.png"])
        if (REPORTS_DIR / "roc_auc.png").exists()
        else None,
    }


@app.route("/media/<str:kind>/<str:filename>", output="bytes", methods="get")
async def media_file(kind: str, filename: str):
    roots = {
        "uploads": MEDIA_UPLOADS_DIR,
        "reports": REPORTS_DIR,
    }
    root = roots.get(kind)
    if root is None:
        response.status = 404
        return b"Not found"

    target = (root / filename).resolve()
    if root.resolve() not in target.parents:
        response.status = 404
        return b"Not found"
    if not target.exists():
        response.status = 404
        return b"Not found"

    response.headers["Content-Type"] = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return target.read_bytes()

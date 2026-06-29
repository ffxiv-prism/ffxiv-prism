from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageFilter, ImageOps

from ffxiv_prism.jsonl import read_jsonl, write_json, write_jsonl
from ffxiv_prism.paths import INDEXES_DIR, PROCESSED_DIR, portable_path, resolve_project_path
from ffxiv_prism.torch_device import resolve_torch_device

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_INDEX_DIR = INDEXES_DIR / "clip-vit-base-patch32"
DEFAULT_DB_PATH = PROCESSED_DIR / "eorzea.sqlite"
VALID_SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")

# Eorzea static PNGs occasionally have a truncated tail but fully decodable pixels.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class RetrievalIndex:
    embeddings: np.ndarray
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]


class ClipImageEncoder:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        self.device = str(resolve_torch_device(device, torch_module=torch))
        self.model_id = model_id
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device)
        except OSError:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with self.torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixel_values)
            if not isinstance(features, self.torch.Tensor):
                features = features.pooler_output
            features = self.torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)

    def encode_paths(self, paths: Sequence[Path], batch_size: int = 16) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            images = [open_rgb(path) for path in paths[start : start + batch_size]]
            try:
                batches.append(self.encode_images(images, batch_size))
            finally:
                for image in images:
                    image.close()
            completed = min(start + batch_size, len(paths))
            batch_number = start // batch_size + 1
            if batch_number % 10 == 0 or completed == len(paths):
                print(f"embedded={completed}/{len(paths)}", flush=True)
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image], batch_size=1)[0]

    def encode_path(self, path: Path) -> np.ndarray:
        image = open_rgb(path)
        try:
            return self.encode_image(image)
        finally:
            image.close()


def choose_device(torch: Any, requested: str) -> str:
    """Backward-compatible wrapper around the shared device resolver."""
    return str(resolve_torch_device(requested, torch_module=torch))


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_candidate_images(
    db_path: Path = DEFAULT_DB_PATH,
    slots: Iterable[str] = VALID_SLOTS,
    include_dyed: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    normalized_slots = validate_slots(slots)
    placeholders = ",".join("?" for _ in normalized_slots)
    sql = f"""
        SELECT
            v.image_row_id, v.source_image_url, v.local_path,
            v.image_slot AS slot, v.image_part, v.model_gender, v.race,
            v.dye_state, v.width, v.height, v.piece_id, v.gearset_slug,
            v.name_en, v.name_zh, v.link_type
        FROM v_piece_images v
        JOIN images i ON i.image_row_id = v.image_row_id
        WHERE i.qa_status = 'ready'
          AND v.image_slot IN ({placeholders})
    """
    params: list[Any] = list(normalized_slots)
    if not include_dyed:
        sql += " AND COALESCE(v.dye_state, 'normal') != 'dyed'"
    sql += " ORDER BY v.image_row_id, v.piece_id"

    grouped: dict[int, dict[str, Any]] = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(sql, params):
            payload = dict(row)
            image_row_id = int(payload["image_row_id"])
            if image_row_id not in grouped:
                grouped[image_row_id] = {
                    key: payload[key]
                    for key in (
                        "image_row_id", "source_image_url", "local_path", "slot",
                        "image_part", "model_gender", "race", "dye_state", "width", "height",
                    )
                }
                grouped[image_row_id]["pieces"] = []
            grouped[image_row_id]["pieces"].append(
                {
                    key: payload[key]
                    for key in ("piece_id", "gearset_slug", "name_en", "name_zh", "link_type")
                }
            )
    finally:
        conn.close()

    rows = list(grouped.values())
    return rows[:limit] if limit is not None else rows


def build_index(
    encoder: ClipImageEncoder,
    rows: Sequence[dict[str, Any]],
    output_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 16,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for row in rows:
        path = resolve_project_path(row["local_path"])
        if path is not None and path.is_file():
            valid_rows.append(row)
            paths.append(path)

    embeddings = encoder.encode_paths(paths, batch_size=batch_size)
    if len(valid_rows) != len(embeddings):
        raise RuntimeError("Embedding row count does not match metadata row count")

    np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    write_jsonl(output_dir / "metadata.jsonl", valid_rows)
    manifest = {
        "version": 1,
        "model_id": encoder.model_id,
        "device": encoder.device,
        "embedding_count": len(valid_rows),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) else 0,
        "embeddings": portable_path(output_dir / "embeddings.npy"),
        "metadata": portable_path(output_dir / "metadata.jsonl"),
        "slots": sorted({row["slot"] for row in valid_rows}),
        "normal_only": all(row.get("dye_state") != "dyed" for row in valid_rows),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> RetrievalIndex:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    if len(metadata) != len(embeddings):
        raise ValueError("Index metadata and embedding counts differ")
    return RetrievalIndex(embeddings=embeddings, metadata=metadata, manifest=manifest)


def search_index(
    index: RetrievalIndex,
    query_embedding: np.ndarray,
    slot: str,
    top_k: int = 10,
    exclude_image_row_id: int | None = None,
    model_gender: str | None = None,
    race: str | None = None,
) -> list[dict[str, Any]]:
    slot = validate_slots([slot])[0]
    query = np.asarray(query_embedding, dtype=np.float32).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    mask = np.fromiter(
        (
            row.get("slot") == slot
            and (exclude_image_row_id is None or row.get("image_row_id") != exclude_image_row_id)
            and (model_gender is None or row.get("model_gender") == model_gender.lower())
            and (race is None or row.get("race") == race.lower())
            for row in index.metadata
        ),
        dtype=bool,
        count=len(index.metadata),
    )
    candidate_indices = np.flatnonzero(mask)
    if not len(candidate_indices):
        return []
    scores = np.asarray(index.embeddings[candidate_indices]) @ query
    image_order = np.argsort(scores)[::-1]

    best_by_piece: dict[int, dict[str, Any]] = {}
    for offset in image_order:
        metadata = index.metadata[int(candidate_indices[offset])]
        score = float(scores[offset])
        for piece in metadata["pieces"]:
            piece_id = int(piece["piece_id"])
            if piece_id in best_by_piece:
                continue
            best_by_piece[piece_id] = {
                "rank": 0,
                "score": score,
                **piece,
                "image_row_id": metadata["image_row_id"],
                "local_path": metadata["local_path"],
                "source_image_url": metadata["source_image_url"],
                "model_gender": metadata.get("model_gender"),
                "race": metadata.get("race"),
                "dye_state": metadata.get("dye_state"),
            }
            if len(best_by_piece) >= top_k:
                break
        if len(best_by_piece) >= top_k:
            break

    results = list(best_by_piece.values())
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def validate_slots(slots: Iterable[str]) -> list[str]:
    normalized = [slot.upper() for slot in slots]
    invalid = sorted(set(normalized) - set(VALID_SLOTS))
    if invalid:
        raise ValueError(f"Invalid slots: {', '.join(invalid)}")
    if not normalized:
        raise ValueError("At least one slot is required")
    return normalized
EVAL_TRANSFORMS = ("mirror_crop", "crop_desaturate")


def transform_eval_image(image: Image.Image, transform: str) -> Image.Image:
    """Apply a deterministic query perturbation without changing outfit identity."""

    if transform not in EVAL_TRANSFORMS:
        raise ValueError(f"Unknown evaluation transform: {transform}")
    width, height = image.size
    inset_x = max(1, round(width * 0.06))
    inset_y = max(1, round(height * 0.06))
    cropped = image.crop((inset_x, inset_y, width - inset_x, height - inset_y))
    if transform == "mirror_crop":
        return ImageOps.mirror(cropped)
    desaturated = ImageEnhance.Color(cropped).enhance(0.72)
    return ImageEnhance.Contrast(desaturated).enhance(1.08)
AUGMENTATION_PROFILES = {
    "mild": (
        "mirror_crop",
        "stretch_wide_mild",
        "squeeze_narrow_mild",
        "rotate_mild",
        "shear_mild",
        "blur_noise_mild",
    ),
    "stress": (
        "mirror_stretch_strong",
        "stretch_wide_strong",
        "squeeze_narrow_strong",
        "rotate_strong",
        "shear_strong",
        "blur_noise_strong",
    ),
}


def augment_eval_image(image: Image.Image, recipe: str, seed: int) -> Image.Image:
    """Create a deterministic, realistically noisy retrieval query image."""

    if recipe not in {name for recipes in AUGMENTATION_PROFILES.values() for name in recipes}:
        raise ValueError(f"Unknown augmentation recipe: {recipe}")
    image = image.convert("RGB")
    width, height = image.size
    background = image.getpixel((0, 0))
    rng = np.random.default_rng(seed)

    if recipe == "mirror_crop":
        return transform_eval_image(image, "mirror_crop")
    if recipe == "stretch_wide_mild":
        return resize_on_canvas(image, 1.10, 1.0, background)
    if recipe == "squeeze_narrow_mild":
        return resize_on_canvas(image, 0.90, 1.0, background)
    if recipe == "rotate_mild":
        return image.rotate(3.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=background)
    if recipe == "shear_mild":
        return shear_image(image, 0.06, background)
    if recipe == "blur_noise_mild":
        return add_blur_noise(image, rng, blur_radius=0.65, noise_sigma=4.0)
    if recipe == "mirror_stretch_strong":
        return resize_on_canvas(ImageOps.mirror(image), 1.22, 0.92, background)
    if recipe == "stretch_wide_strong":
        return resize_on_canvas(image, 1.25, 0.90, background)
    if recipe == "squeeze_narrow_strong":
        return resize_on_canvas(image, 0.76, 1.08, background)
    if recipe == "rotate_strong":
        return image.rotate(-7.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=background)
    if recipe == "shear_strong":
        return shear_image(image, -0.14, background)
    return add_blur_noise(image, rng, blur_radius=1.35, noise_sigma=10.0)


def resize_on_canvas(
    image: Image.Image,
    scale_x: float,
    scale_y: float,
    background: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    resized = image.resize(
        (max(1, round(width * scale_x)), max(1, round(height * scale_y))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (width, height), background)
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    resized.close()
    return canvas


def shear_image(
    image: Image.Image,
    shear: float,
    background: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    offset = -shear * height / 2
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1.0, shear, offset, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=background,
    )


def add_blur_noise(
    image: Image.Image,
    rng: np.random.Generator,
    blur_radius: float,
    noise_sigma: float,
) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    pixels = np.asarray(blurred, dtype=np.float32)
    noise = rng.normal(0.0, noise_sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    blurred.close()
    return Image.fromarray(noisy, mode="RGB")

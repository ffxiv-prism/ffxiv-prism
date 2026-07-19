from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from ffxiv_prism.vision.retrieval import (
    RetrievalIndex,
    augment_eval_image,
    search_index,
    select_primary_visual_pieces,
    transform_eval_image,
    validate_slots,
)


class VisualRetrievalTests(unittest.TestCase):
    def test_search_filters_slot_and_dedupes_piece(self) -> None:
        index = RetrievalIndex(
            embeddings=np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]], dtype=np.float32),
            metadata=[image(1, "BODY", [10]), image(2, "BODY", [10, 11]), image(3, "HEAD", [12])],
            manifest={"model_id": "test"},
        )
        results = search_index(index, np.asarray([1.0, 0.0]), "BODY", top_k=5)
        self.assertEqual([row["piece_id"] for row in results], [10, 11])
        self.assertEqual(results[0]["image_row_id"], 1)

    def test_search_filters_model_gender(self) -> None:
        female = image(1, "BODY", [10])
        female["model_gender"] = "female"
        male = image(2, "BODY", [11])
        male["model_gender"] = "male"
        index = RetrievalIndex(
            embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            metadata=[female, male],
            manifest={"model_id": "test"},
        )
        results = search_index(index, np.asarray([1.0, 0.0]), "BODY", model_gender="male")
        self.assertEqual([row["piece_id"] for row in results], [11])

    def test_primary_visual_piece_filter_removes_unrelated_set_items(self) -> None:
        pieces = [
            {"piece_id": 1, "name_en": "Historia Shoes of Casting"},
            {"piece_id": 2, "name_en": "Historia Earrings of Casting"},
            {"piece_id": 3, "name_en": "Historia Staff"},
            {"piece_id": 4, "name_en": "Augmented Historia Shoes of Casting"},
        ]
        selected = select_primary_visual_pieces(pieces)
        self.assertEqual([piece["piece_id"] for piece in selected], [1, 4])

    def test_primary_visual_piece_filter_keeps_single_piece(self) -> None:
        pieces = [{"piece_id": 7, "name_en": "High House Boots"}]
        self.assertEqual(select_primary_visual_pieces(pieces), pieces)

    def test_search_can_exclude_query_image(self) -> None:
        index = RetrievalIndex(
            embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
            metadata=[image(1, "FEET", [20]), image(2, "FEET", [20])],
            manifest={"model_id": "test"},
        )
        results = search_index(index, np.asarray([1.0, 0.0]), "FEET", top_k=1, exclude_image_row_id=1)
        self.assertEqual(results[0]["image_row_id"], 2)

    def test_augmentation_is_deterministic_and_preserves_canvas(self) -> None:
        image = Image.new("RGB", (100, 200), (240, 220, 200))
        first = augment_eval_image(image, "blur_noise_strong", seed=42)
        second = augment_eval_image(image, "blur_noise_strong", seed=42)
        try:
            self.assertEqual(first.size, image.size)
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertNotEqual(first.tobytes(), image.tobytes())
        finally:
            first.close()
            second.close()

    def test_stretch_augmentation_preserves_canvas(self) -> None:
        image = Image.new("RGB", (100, 200), "white")
        stretched = augment_eval_image(image, "stretch_wide_strong", seed=1)
        try:
            self.assertEqual(stretched.size, image.size)
        finally:
            stretched.close()
    def test_eval_transform_preserves_image_mode_and_changes_size(self) -> None:
        image = Image.new("RGB", (100, 200), "red")
        transformed = transform_eval_image(image, "mirror_crop")
        try:
            self.assertEqual(transformed.mode, "RGB")
            self.assertEqual(transformed.size, (88, 176))
        finally:
            transformed.close()
    def test_validate_slots_rejects_unknown_slot(self) -> None:
        with self.assertRaises(ValueError):
            validate_slots(["weapon"])


def image(image_row_id: int, slot: str, piece_ids: list[int]) -> dict:
    return {
        "image_row_id": image_row_id,
        "slot": slot,
        "local_path": f"image-{image_row_id}.png",
        "source_image_url": f"https://example.test/{image_row_id}.png",
        "pieces": [
            {
                "piece_id": piece_id,
                "gearset_slug": "test",
                "name_en": f"Piece {piece_id}",
                "name_zh": None,
                "link_type": "exact_slot",
            }
            for piece_id in piece_ids
        ],
    }


if __name__ == "__main__":
    unittest.main()

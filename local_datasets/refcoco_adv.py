from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
def coco_to_xyxy(bbox_coco, width, height):
    # Keep unchanged
    x, y, w, h = bbox_coco
    if w <= 0 or h <= 0: return None
    x1=max(0, float(x)); y1=max(0, float(y))
    x2=min(float(width), float(x) + float(w)); y2=min(float(height), float(y) + float(h))
    # Return as integers or floats? IoU usually uses floats. Keep as floats for precision.
    return [x1, y1, x2, y2]

class RefCOCOAdvDataset:

    # ────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────
    def __init__(
        self,
        annotation_file: str | os.PathLike,
        image_dir: str | os.PathLike,
        num_samples: Optional[int] = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load JSON (dictionary mapping unique_id -> sample‑dict)
        with self.annotation_file.open() as f:
            raw_data: Dict[str, Dict[str, Any]] = json.load(f)

        entries: List[Tuple[str, Dict[str, Any]]] = list(raw_data.items())

        # Optional truncation
        if num_samples is not None:
            entries = entries[: num_samples]

        # Store parallel lists for deterministic ordering & fast lookup
        self._ids: List[str] = [uid for uid, _ in entries]
        self._items: List[Dict[str, Any]] = [itm for _, itm in entries]

    # ────────────────────────────────────────────────
    # Python data‑model dunder methods
    # ────────────────────────────────────────────────
    def __len__(self) -> int:  # noqa: D401
        """Return the number of *samples* in the dataset."""
        return len(self._items)

    # Utility: parse bbox map safely once
    @staticmethod
    def _parse_bbox_map(bbox_map_str: str) -> Dict[str, List[float]]:
        try:
            bbox_map = json.loads(bbox_map_str)
            if not isinstance(bbox_map, dict):
                raise TypeError("bbox_candidates_map must decode to a dict")
            for k, v in bbox_map.items():
                if not (isinstance(v, list) and len(v) == 4):
                    raise ValueError(f"bbox for key {k} is not length‑4 list: {v}")
            return bbox_map
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Could not parse bbox_candidates_map: {exc}") from exc

    def _resolve_image_path(self, img_file_name: str) -> Path:
        """Return *absolute* path to a COCO image file."""
        if "train2014" in img_file_name:
            subdir = "train2014"
        elif "val2014" in img_file_name:
            subdir = "val2014"
        else:
            subdir = ""
        return self.image_dir / subdir / img_file_name if subdir else self.image_dir / img_file_name

    def __getitem__(self, index: int) -> Dict[str, Any]:  # noqa: D401
        """Return a dict with all metadata for *one* sample.

        The returned structure purposefully mirrors the keys expected by the
        evaluation script so that you can drop this class in without further
        refactors.
        """
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of range [0, {len(self) - 1}]")

        uid: str = self._ids[index]
        itm: Dict[str, Any] = self._items[index]

        img_file_name: str = itm["img_file_name"]
        image_path: Path = self._resolve_image_path(img_file_name)

        # Bounding boxes
        bbox_map: Dict[str, List[float]] = self._parse_bbox_map(itm["bbox_candidates_map"])
        gt_id: str = itm["most_confused_bbox_number"]
        if gt_id not in bbox_map:
            raise KeyError(f"GT bbox id {gt_id!r} missing for sample {uid}")
        image = Image.open(image_path).convert("RGB")
        return {
            'id': uid,
            'query': itm["most_confused_bbox_GT_desc"],
            'answer': coco_to_xyxy(bbox_map[gt_id], image.width, image.height),
            'image': image,
            'image_path': img_file_name,
        }

    # ────────────────────────────────────────────────
    # Convenience helpers (do NOT get called implicitly)
    # ────────────────────────────────────────────────
    def load_image(self, index: int, *, mode: str = "RGB") -> Image.Image:  # noqa: D401
        """Lazily load a PIL image for *one* sample (converts to RGB by default)."""
        img_path = self[index]["image_path"]
        return Image.open(img_path).convert(mode)

    def get_sample_by_id(self, unique_id: str) -> Dict[str, Any]:
        """Fast lookup by unique_id (O(1))."""
        try:
            idx = self._ids.index(unique_id)
        except ValueError as exc:
            raise KeyError(f"unique_id {unique_id!r} not in dataset") from exc
        return self[idx]

    # ────────────────────────────────────────────────
    # Representation helpers
    # ────────────────────────────────────────────────
    def __repr__(self) -> str:  # noqa: D401
        split_str = f" split={self.split!r}" if self.split else ""
        return (
            f"{self.__class__.__name__}(n={len(self)}{split_str}, "
            f"annotation_file={self.annotation_file.name!r})"
        )

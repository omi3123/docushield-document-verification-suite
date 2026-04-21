from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import fitz
import cv2
import numpy as np
from PIL import Image

def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def cv_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def load_document_pages(file_path: str) -> List[Tuple[int, Image.Image]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(file_path)
        pages = []
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append((idx + 1, img))
        return pages
    return [(1, Image.open(file_path).convert("RGB"))]

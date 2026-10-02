import base64
import io
from typing import List, Dict, Any

import fitz
from PIL import Image


def process_drawing(pdf_bytes: bytes, max_pages: int = 10) -> List[Dict[str, Any]]:
    """Render PDF pages into reasonably sized JPEG images for vision analysis."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    try:
        total = min(len(doc), max_pages)

        for page_no in range(total):
            page = doc.load_page(page_no)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)

            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            max_side = 1800
            scale = min(1.0, max_side / max(image.size))
            if scale < 1.0:
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale))
                )

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)

            pages.append(
                {
                    "page_number": page_no + 1,
                    "mime_type": "image/jpeg",
                    "image_bytes": buffer.getvalue(),
                    "image_b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                }
            )
    finally:
        doc.close()

    return pages

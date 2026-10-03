import base64
import io
import re
from typing import Any, Dict, List

import fitz  # PyMuPDF
from PIL import Image

MAX_SIDE = 1600                 # longest image side in pixels
MAX_B64_BYTES_PER_PAGE = 1_100_000  # keeps several pages well under Groq's request-size limits
MAX_HINTS = 45

# Dimension-like or label-like text lines found in the PDF text layer (vector PDFs only).
_DIM_RE = re.compile(
    r"""^(?:
        \d{1,2}[.,]\d{1,3}\s?(?:mm|cm|m)?          # 4.20  4,2 m
        |\d{3,5}\s?(?:mm|cm|m)?                    # 3600  3600mm
        |\d+(?:[.,]\d+)?\s?[xX×]\s?\d+(?:[.,]\d+)?\s?(?:mm|cm|m)?   # 4.2 x 3.6
        |\d+'\s?-?\s?\d*(?:\s?\d+/\d+)?"?          # 12'-6"
        |1\s?:\s?\d{1,4}                           # 1:100
        |scale\b.{0,20}
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
_LABEL_RE = re.compile(
    r"bed|bath|kitchen|living|dining|hall|corridor|toilet|\bwc\b|store|office|lobby|stair|"
    r"balcony|garage|utility|lounge|room|ground floor|first floor|level|plan|elevation|section",
    re.IGNORECASE,
)


def extract_text_hints(text: str, limit: int = MAX_HINTS) -> List[str]:
    """Pick short dimension/label strings from a page's text layer. No OCR, no AI."""
    seen, hints = set(), []
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if not line or len(line) > 28 or line.lower() in seen:
            continue
        if _DIM_RE.match(line) or _LABEL_RE.search(line):
            seen.add(line.lower())
            hints.append(line)
            if len(hints) >= limit:
                break
    return hints


def _encode_jpeg(image: Image.Image) -> bytes:
    """JPEG-encode, lowering quality/size until the base64 payload fits the per-page budget."""
    quality = 80
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) * 4 // 3 <= MAX_B64_BYTES_PER_PAGE or max(image.size) <= 600:
            return data
        if quality > 55:
            quality -= 12
        else:
            image = image.resize((int(image.width * 0.85), int(image.height * 0.85)))


def process_drawing(pdf_bytes: bytes, max_pages: int = 3) -> List[Dict[str, Any]]:
    """Render PDF pages to compact JPEGs for vision analysis and collect text-layer hints."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Could not open the file as a PDF.") from exc

    pages: List[Dict[str, Any]] = []
    try:
        if doc.needs_pass:
            raise ValueError("The PDF is password-protected.")
        if len(doc) == 0:
            raise ValueError("The PDF has no pages.")

        for page_no in range(min(len(doc), max_pages)):
            page = doc.load_page(page_no)

            # Render straight to the target size so huge CAD sheets never allocate huge bitmaps.
            longest = max(page.rect.width, page.rect.height) or 1.0
            zoom = max(0.3, min(3.0, MAX_SIDE / longest))
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            data = _encode_jpeg(image)
            pages.append(
                {
                    "page_number": page_no + 1,
                    "mime_type": "image/jpeg",
                    "image_bytes": data,
                    "image_b64": base64.b64encode(data).decode("utf-8"),
                    "text_hints": extract_text_hints(page.get_text("text")),
                }
            )
    finally:
        doc.close()

    return pages

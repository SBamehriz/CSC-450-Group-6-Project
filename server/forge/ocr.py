import io
import os
import tempfile
from pathlib import Path

from PIL import Image

# Suppress harmless Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")

_docling_converter = None
_rapidocr = None
_pytesseract = None


def _get_docling_converter():
    """Lazily load and return Docling DocumentConverter singleton if installed."""
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.document_converter import DocumentConverter

            _docling_converter = DocumentConverter()
        except ImportError:
            _docling_converter = False
    return _docling_converter if _docling_converter is not False else None


def _get_rapidocr():
    """Lazily load RapidOCR engine if installed."""
    global _rapidocr
    if _rapidocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _rapidocr = RapidOCR()
        except ImportError:
            try:
                from rapidocr import RapidOCR

                _rapidocr = RapidOCR()
            except ImportError:
                _rapidocr = False
    return _rapidocr if _rapidocr is not False else None


def _get_pytesseract():
    """Lazily load pytesseract if installed."""
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract

            _pytesseract = pytesseract
        except ImportError:
            _pytesseract = False
    return _pytesseract if _pytesseract is not False else None


def is_ocr_available() -> bool:
    """Return True if at least one OCR engine (Docling, RapidOCR, Tesseract) is available."""
    return bool(_get_docling_converter() or _get_rapidocr() or _get_pytesseract())


def get_ocr_engine_name() -> str:
    """Return the name of the primary active OCR engine."""
    if _get_docling_converter() is not None:
        return "Docling (RapidOCR)"
    if _get_rapidocr() is not None:
        return "RapidOCR"
    if _get_pytesseract() is not None:
        return "PyTesseract"
    return "None"


def ocr_image(
    image_input: bytes | Path | str | Image.Image,
    filename: str = "image.png",
) -> str:
    """Perform optical character recognition (OCR) on an image file or bytes.

    Supports PNG, JPG, JPEG, TIFF, BMP, and WebP formats. Preprocesses the image
    to RGB and dispatches to the available OCR engine (RapidOCR, Docling, or PyTesseract).
    """
    raw_bytes: bytes
    img: Image.Image

    if isinstance(image_input, Image.Image):
        img = image_input
        buf = io.BytesIO()
        # Save as PNG to preserve fidelity for in-memory buffer
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        raw_bytes = path.read_bytes()
        img = Image.open(io.BytesIO(raw_bytes))
        filename = path.name
    else:
        raw_bytes = image_input
        img = Image.open(io.BytesIO(raw_bytes))

    # Preprocess image: ensure RGB mode (normalizes RGBA transparency, palette, 1-bit, CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Strategy 1: Direct RapidOCR execution if available
    rapid = _get_rapidocr()
    if rapid is not None:
        try:
            import numpy as np

            result, _ = rapid(np.array(img))
            if result:
                lines = [line[1] for line in result if line and len(line) > 1 and line[1]]
                text = "\n".join(lines).strip()
                if text:
                    return text
        except Exception:
            pass

    # Strategy 2: Docling DocumentConverter execution (built-in OCR & layout)
    docling = _get_docling_converter()
    if docling is not None:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            ext = ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = docling.convert(str(tmp_path))
            text = result.document.export_to_markdown().strip()
            if text:
                return text
        finally:
            tmp_path.unlink(missing_ok=True)

    # Strategy 3: PyTesseract execution
    tess = _get_pytesseract()
    if tess is not None:
        try:
            text = tess.image_to_string(img).strip()
            if text:
                return text
        except Exception:
            pass

    if not is_ocr_available():
        raise RuntimeError(
            "No OCR engine is installed. Please install 'docling' or 'rapidocr-onnxruntime' "
            "via 'uv sync --extra converter' or 'pip install docling rapidocr-onnxruntime'."
        )

    return ""


def ocr_pdf(
    pdf_input: bytes | Path | str,
    filename: str = "document.pdf",
) -> str:
    """Perform optical character recognition (OCR) on a scanned PDF document.

    Dispatches to Docling's OCR pipeline, or extracts embedded page images via pypdf
    and applies image OCR page-by-page.
    """
    raw_bytes: bytes
    if isinstance(pdf_input, (str, Path)):
        path = Path(pdf_input)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        raw_bytes = path.read_bytes()
        filename = path.name
    else:
        raw_bytes = pdf_input

    # Strategy 1: Docling DocumentConverter (extracts text, tables, and OCR on scanned pages)
    docling = _get_docling_converter()
    if docling is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = docling.convert(str(tmp_path))
            text = result.document.export_to_markdown().strip()
            if text:
                return text
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)

    # Strategy 2: Extract embedded raster images from PDF pages with pypdf and run OCR
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        page_texts: list[str] = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_extracted = []
            for img in getattr(page, "images", []):
                try:
                    img_text = ocr_image(img.data, filename=f"{filename}_p{page_num}_{img.name}")
                    if img_text.strip():
                        page_extracted.append(img_text.strip())
                except Exception:
                    continue
            if page_extracted:
                page_texts.append("\n\n".join(page_extracted))

        if page_texts:
            return "\n\n".join(page_texts).strip()
    except Exception:
        pass

    if not is_ocr_available():
        raise RuntimeError(
            "No OCR engine is installed. Please install 'docling' or 'rapidocr-onnxruntime' "
            "via 'uv sync --extra converter' or 'pip install docling rapidocr-onnxruntime'."
        )

    return ""


def ocr_document(
    file_input: bytes | Path | str,
    filename: str = "",
) -> str:
    """Unified OCR dispatcher for scanned PDFs and PNG/JPG/JPEG images."""
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not filename:
            filename = path.name
        ext = path.suffix.lower()
        raw = path.read_bytes()
    else:
        raw = file_input
        ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return ocr_pdf(raw, filename=filename or "document.pdf")
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return ocr_image(raw, filename=filename or "image.png")

    raise ValueError(
        f"Unsupported format for OCR: {ext}. "
        f"Supported: .pdf, {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
    )

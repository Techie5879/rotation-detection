"""PDF rendering and metadata-neutralization utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterator

from .constants import WHITE_RGB


def _pil_to_rgb(image):
    """Convert PIL image modes into RGB with white background."""
    from PIL import Image

    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        base = Image.new("RGB", image.size, WHITE_RGB)
        alpha = image.split()[-1]
        base.paste(image.convert("RGB"), mask=alpha)
        return base
    return image.convert("RGB")


def rotate_image_clockwise(image, angle: int):
    """Rotate an image clockwise by a cardinal angle and expand canvas."""
    from PIL import Image

    if angle % 360 == 0:
        return _pil_to_rgb(image)
    if angle % 360 not in {90, 180, 270}:
        raise ValueError(f"Only cardinal clockwise rotation is supported, got {angle}.")
    return _pil_to_rgb(
        image.rotate(
            -angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=WHITE_RGB,
        )
    )


def iter_rendered_pages(pdf_path: Path, dpi: int = 144) -> Iterator[tuple[int, object]]:
    """Yield rendered PIL images for each page in the PDF."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=scale, rotation=0)
            image = _pil_to_rgb(bitmap.to_pil())
            yield page_index, image
    finally:
        pdf.close()


def append_image_as_pdf_page(writer, image, dpi: int = 144) -> None:
    """Append a single PIL image as a one-page PDF into a PdfWriter."""
    from pypdf import PdfReader

    page_buffer = BytesIO()
    _pil_to_rgb(image).save(page_buffer, format="PDF", resolution=dpi)
    page_buffer.seek(0)
    image_pdf = PdfReader(page_buffer)
    page = image_pdf.pages[0]
    if "/Rotate" in page:
        del page["/Rotate"]
    writer.add_page(page)


def strip_page_rotation_metadata(input_pdf: Path, output_pdf: Path) -> None:
    """Write a PDF copy with page /Rotate metadata removed."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()

    for page in reader.pages:
        if "/Rotate" in page:
            del page["/Rotate"]
        writer.add_page(page)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def rotate_metadata_violations(pdf_path: Path) -> list[dict[str, int]]:
    """Return pages that still expose non-zero /Rotate metadata."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    violations: list[dict[str, int]] = []
    for page_index, page in enumerate(reader.pages):
        page_rotate = int(page.get("/Rotate", 0) or 0) % 360
        if page_rotate != 0:
            violations.append({"page_index": page_index, "rotate": page_rotate})
    return violations

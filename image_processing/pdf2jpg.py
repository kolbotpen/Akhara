"""
PDF to JPG Converter
====================
Converts each page of one or more PDF files to a JPG image.

Uses pymupdf (fitz) as the primary engine — no system dependencies needed.
Falls back to pdf2image (poppler) if pymupdf is not installed.

Output naming:
    single-page PDF  ->  <stem>.jpg
    multi-page PDF   ->  <stem>_page1.jpg, <stem>_page2.jpg, ...

Usage:
    python pdf_to_jpg.py document.pdf
    python pdf_to_jpg.py pdfs_to_print/*.pdf --output-dir ./image_processing/images
    python pdf_to_jpg.py document.pdf --dpi 200 --quality 85

Requirements (pick one):
    pip install pymupdf            <- recommended, no system deps
    pip install pdf2image pillow   <- also needs: apt install poppler-utils
"""

import argparse
import sys
from pathlib import Path


# ── Conversion engines ────────────────────────────────────────────────────────

def convert_with_pymupdf(pdf_path: Path, output_dir: Path, dpi: int, quality: int):
    import fitz  # pymupdf
    doc   = fitz.open(str(pdf_path))
    n     = len(doc)
    saved = []
    mat   = fitz.Matrix(dpi / 72, dpi / 72)   # 72 pt/inch is PDF native

    for i, page in enumerate(doc):
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        out_name = f"{pdf_path.stem}.jpg" if n == 1 else f"{pdf_path.stem}_page{i+1}.jpg"
        out_path = output_dir / out_name
        pix.save(str(out_path))                # pymupdf writes JPEG directly
        saved.append(out_path)
        print(f"  ✓  page {i+1}/{n}  ->  {out_name}")

    doc.close()
    return saved


def convert_with_pdf2image(pdf_path: Path, output_dir: Path, dpi: int, quality: int):
    from pdf2image import convert_from_path
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    n     = len(pages)
    saved = []

    for i, page in enumerate(pages):
        out_name = f"{pdf_path.stem}.jpg" if n == 1 else f"{pdf_path.stem}_page{i+1}.jpg"
        out_path = output_dir / out_name
        page.save(str(out_path), "JPEG", quality=quality)
        saved.append(out_path)
        print(f"  ✓  page {i+1}/{n}  ->  {out_name}")

    return saved


def convert_pdf(pdf_path: Path, output_dir: Path, dpi: int, quality: int):
    print(f"\n── {pdf_path.name}  ({dpi} DPI)")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try pymupdf first (no system deps), then pdf2image
    try:
        import fitz
        return convert_with_pymupdf(pdf_path, output_dir, dpi, quality)
    except ImportError:
        pass
    except Exception as e:
        print(f"  ✗ pymupdf error: {e}")
        sys.exit(1)

    try:
        from pdf2image import convert_from_path
        return convert_with_pdf2image(pdf_path, output_dir, dpi, quality)
    except ImportError:
        print("  ✗ No PDF engine found. Install one:")
        print("      pip install pymupdf")
        print("      -- or --")
        print("      pip install pdf2image pillow && apt install poppler-utils")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ pdf2image error: {e}")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert PDF pages to JPG images.")
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to convert")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output folder (default: same folder as the PDF)")
    parser.add_argument("--dpi",     type=int, default=300,
                        help="Render resolution (default: 300)")
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality 1-95 (default: 90, pymupdf ignores this)")
    args = parser.parse_args()

    all_outputs = []
    for pattern in args.pdfs:
        matches = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        for pdf_path in matches:
            if not pdf_path.exists():
                print(f"⚠  Not found: {pdf_path}")
                continue
            if pdf_path.suffix.lower() != ".pdf":
                print(f"⚠  Skipping non-PDF: {pdf_path.name}")
                continue
            out_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent
            all_outputs += convert_pdf(pdf_path, out_dir, args.dpi, args.quality)

    print(f"\n{'='*45}")
    print(f"  Done. {len(all_outputs)} page(s) saved.")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
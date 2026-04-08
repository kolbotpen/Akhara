"""
Handwriting Box Extractor
=========================
Detects handwriting boxes in scanned pages, reads the ID number printed
above-left of each box via OCR, crops the interior (no border), resizes
to exactly 960x384 px, saves as <word_id>.png, and appends a row to
dataset.csv.

dataset.csv columns:
    id          - auto-incrementing integer (continues across runs)
    word_id     - the number printed on the page next to the box
    image_path  - path to the saved crop
    writer_id   - auto-incremented per image file (1 image = 1 writer)
                  override the starting number with --writer-id-start
    label       - Khmer text from word_labels.csv (blank if not found)

Usage:
    python extract_boxes.py
    python extract_boxes.py --debug
    python extract_boxes.py --writer-id-start 3   # if adding more writers later

Requirements:
    pip install opencv-python pytesseract pandas numpy
    sudo apt-get install tesseract-ocr
    (macOS: brew install tesseract)
"""

import cv2
import numpy as np
import csv
import re
import argparse
from pathlib import Path
from collections import Counter

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠  pytesseract not installed.")
    print("   pip install pytesseract  +  apt/brew install tesseract-ocr")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_W        = 960
OUT_H        = 384
BORDER_INSET = 4   # px to inset from detected box edge to exclude border line


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(labels_csv):
    """Return {word_id (int) -> khmer_text (str)} from word_labels.csv."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    # word_id may appear multiple times (same word on multiple PDFs) — just
    # take the first occurrence; the text is the same for identical word_ids.
    return {int(row['word_id']): str(row['word_text'])
            for _, row in df.drop_duplicates('word_id').iterrows()}


# ── Dataset CSV helpers ───────────────────────────────────────────────────────

DATASET_FIELDS = ['id', 'image_path', 'writer_id', 'label']


def next_auto_id(dataset_csv: Path) -> int:
    """Return the next auto-increment id (max existing id + 1, or 1 if new)."""
    if not dataset_csv.exists():
        return 1
    max_id = 0
    with open(dataset_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row['id']))
            except (ValueError, KeyError):
                pass
    return max_id + 1


def append_rows(dataset_csv: Path, rows: list):
    """Append rows to dataset CSV, writing header if file is new."""
    file_exists = dataset_csv.exists()
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Box detection ─────────────────────────────────────────────────────────────

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix  = max(ax, bx);       iy  = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0


def _merge_rects(rects, iou_thresh=0.35):
    if not rects:
        return rects
    rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    keep = []
    for r in rects:
        if all(_iou(r, k) < iou_thresh for k in keep):
            keep.append(r)
    return keep


def find_boxes(img_bgr):
    """Find all handwriting rectangles, sorted left-col top->bottom then right."""
    h, w  = img_bgr.shape[:2]
    total = h * w
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(adaptive, otsu)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area   = bw * bh
        aspect = bw / bh if bh else 0
        if (0.004 * total < area < 0.20 * total) and (1.4 < aspect < 7.0):
            candidates.append((x, y, bw, bh))

    candidates = _merge_rects(candidates)
    mid   = w / 2
    left  = sorted([r for r in candidates if r[0]+r[2]/2 <  mid], key=lambda r: r[1])
    right = sorted([r for r in candidates if r[0]+r[2]/2 >= mid], key=lambda r: r[1])
    return left + right


# ── OCR: read ID above the box ────────────────────────────────────────────────

def _ocr_number(gray_roi):
    """Try multiple preprocessing + PSM combos, majority-vote the result."""
    hits = []
    for scale in [2, 3, 4]:
        up = cv2.resize(gray_roi, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
        _, th     = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv    = cv2.bitwise_not(th)
        for variant in [th, th_inv]:
            padded = cv2.copyMakeBorder(variant, 20, 20, 20, 20,
                                        cv2.BORDER_CONSTANT, value=255)
            for psm in [7, 8, 6]:
                cfg  = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
                try:
                    text = pytesseract.image_to_string(padded, config=cfg).strip()
                    m    = re.search(r'\d+', text)
                    if m:
                        hits.append(int(m.group()))
                except Exception:
                    pass
    return Counter(hits).most_common(1)[0][0] if hits else None


def read_id_above_box(img_bgr, box):
    """OCR the number printed above-left of the box."""
    if not OCR_AVAILABLE:
        return None
    x, y, w, h   = box
    ih, iw        = img_bgr.shape[:2]
    x1 = max(0,  x - 10)
    x2 = min(iw, x + w)
    y1 = max(0,  y - int(h * 1.2))
    y2 = max(0,  y - BORDER_INSET)
    if y2 <= y1 or x2 <= x1:
        return None

    roi  = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Try left-third first (where the ID number sits), then full strip
    left = gray[:, : max(1, gray.shape[1] // 3)]
    return _ocr_number(left) or _ocr_number(gray)


# ── Crop + resize ─────────────────────────────────────────────────────────────

def crop_box(img_bgr, box):
    x, y, w, h = box
    ih, iw = img_bgr.shape[:2]
    x1 = max(0,  x + BORDER_INSET)
    y1 = max(0,  y + BORDER_INSET)
    x2 = min(iw, x + w - BORDER_INSET)
    y2 = min(ih, y + h - BORDER_INSET)
    crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(img_path, output_dir, writer_id, labels,
                  dataset_csv, debug=False):
    """
    Process one scanned image. Appends to dataset_csv immediately after
    each image so partial runs are not lost.
    Returns number of crops saved.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Cannot read: {img_path.name}")
        return 0

    print(f"\n── {img_path.name}  (writer_id={writer_id})")
    boxes = find_boxes(img)
    print(f"   Found {len(boxes)} box(es)")
    if not boxes:
        print("   ⚠  No boxes detected — skipping.")
        return 0

    # Get current auto-increment base before we start writing
    auto_id   = next_auto_id(dataset_csv)
    saved     = 0
    new_rows  = []
    debug_img = img.copy() if debug else None

    for i, box in enumerate(boxes):
        word_id = read_id_above_box(img, box)
        if word_id is None:
            print(f"   ⚠  box {i+1}: OCR could not read ID — skipping.")
            continue

        resized  = crop_box(img, box)
        out_name = f"{word_id}_{writer_id}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), resized)

        label = labels.get(word_id, '')
        row   = {
            'id':         auto_id,
            'image_path': str(out_path),
            'writer_id':  writer_id,
            'label':      label,
        }
        new_rows.append(row)
        auto_id += 1
        saved   += 1

        print(f"   ✓  box {i+1}: word_id={word_id}  label=\"{label}\"  -> {out_name}")

        if debug:
            x, y, w, h = box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 200, 0), 3)
            cv2.putText(debug_img, str(word_id), (x+6, y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

    # Append this image's rows immediately
    if new_rows:
        append_rows(dataset_csv, new_rows)

    if debug and debug_img is not None:
        dbg_path = output_dir / f"DEBUG_{img_path.stem}.jpg"
        cv2.imwrite(str(dbg_path), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"   Debug -> {dbg_path.name}")

    return saved


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f'Crop handwriting boxes ({OUT_W}x{OUT_H}px) and build dataset CSV.'
    )
    parser.add_argument('--images-dir',      default='./image_processing/images',
                        help='Folder with scanned images')
    parser.add_argument('--output-dir',      default='./image_processing/processed_images',
                        help='Where to save cropped images')
    parser.add_argument('--labels-csv',      default='./pdfs_to_print/word_labels.csv',
                        help='word_labels.csv from generate_pdf.py')
    parser.add_argument('--dataset-csv',     default='./image_processing/dataset.csv',
                        help='Dataset CSV to create or append to')
    parser.add_argument('--writer-id-start', default=1, type=int,
                        help='First writer_id to assign (increments per image file)')
    parser.add_argument('--debug',           action='store_true',
                        help='Save annotated debug image per page')
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    images_dir  = Path(args.images_dir)
    dataset_csv = Path(args.dataset_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels (gracefully handle missing file)
    labels      = {}
    labels_path = Path(args.labels_csv)
    if labels_path.exists():
        labels = load_labels(labels_path)
        print(f"✓ Loaded {len(labels)} labels from {labels_path.name}")
    else:
        print(f"⚠  word_labels.csv not found at {labels_path} — label column will be blank")

    # Find images
    exts   = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    images = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in exts and not f.name.startswith('DEBUG_')
    ])
    if not images:
        print(f"✗ No images found in {images_dir}")
        return

    print(f"✓ {len(images)} image(s) found")
    print(f"  Output size : {OUT_W}x{OUT_H}px")
    print(f"  Dataset CSV : {dataset_csv}  (appending)")
    print(f"  writer_id   : {args.writer_id_start} .. {args.writer_id_start + len(images) - 1}")

    total_saved = 0
    for i, img_path in enumerate(images):
        writer_id    = args.writer_id_start + i
        total_saved += process_image(
            img_path, output_dir,
            writer_id   = writer_id,
            labels      = labels,
            dataset_csv = dataset_csv,
            debug       = args.debug,
        )

    print(f"\n{'='*52}")
    print(f"  Done. {total_saved} crops saved.")
    print(f"  Images  -> {output_dir}/")
    print(f"  Dataset -> {dataset_csv}")
    print(f"{'='*52}")
    print()
    print("To add more writers later:")
    print("  python extract_boxes.py --writer-id-start 3")
    print("  (id column auto-continues from last row in dataset.csv)")


if __name__ == '__main__':
    main()
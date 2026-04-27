"""
PDF Generator for Labeled Handwriting Data
Generates PDFs with format: "ID word" (one per line)
"""
import os
import csv
import random
import argparse
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image, ImageDraw, ImageFont
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
import pandas as pd
import sys
import io


def check_raqm_support():
    """Check if PIL has RAQM support for complex text shaping"""
    try:
        features = ImageFont.core.freetype2_version
        # Try to access RAQM layout
        test_img = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(test_img)
        try:
            # This will fail if RAQM is not available
            ImageFont.Layout.RAQM
            print("✓ RAQM text shaping engine is available (good for Khmer)")
            return True
        except AttributeError:
            print("⚠ RAQM not available. Install libraqm for better Khmer rendering:")
            print("  macOS: brew install libraqm")
            print("  Linux: sudo apt-get install libraqm")
            print("  Then: pip install --upgrade pillow")
            return False
    except:
        return False


def load_word_list(word_list_path):
    # load word list from csv file
    words = {}
    with open(word_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            words[int(row['id'])] = row['khmer_text']
    return words


def sample_words(words, count):
    """Return a list of (word_id, word_text) pairs for one page."""
    word_ids = list(words.keys())
    if count <= len(word_ids):
        chosen_ids = random.sample(word_ids, count)
    else:
        chosen_ids = random.choices(word_ids, k=count)
    return [(wid, words[wid]) for wid in chosen_ids]


def find_khmer_font():
    """
    Find a Khmer-compatible font for PIL rendering.
    Returns font path or None.
    """
    font_paths = [
        "fonts/NotoSansKhmer-Regular.ttf",
        "./fonts/NotoSansKhmer-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSansKhmer-Regular.ttf",
        "/Library/Fonts/Noto Sans Khmer.ttf",
        "C:\\Windows\\Fonts\\NotoSansKhmer-Regular.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"✓ Found Khmer font: {font_path}")
            return font_path
    
    print("\n" + "="*60)
    print("⚠ WARNING: No suitable Khmer .ttf font found!")
    print("="*60)
    print("Text will not display correctly without a proper Khmer font.")
    print("\nQuick fix:")
    print("  Run: python download_khmer_font.py")
    print("\nOr manually:")
    print("  1. Download Noto Sans Khmer from Google Fonts")
    print("  2. Save as: fonts/NotoSansKhmer-Regular.ttf")
    print("="*60 + "\n")
    return None


def render_text_as_image(text, font_size=6, font_path=None):
    """
    Render text as an image using PIL (properly handles Khmer Unicode).
    
    Args:
        text: Text to render
        font_size: Font size in points
        font_path: Path to TrueType font file
    
    Returns:
        PIL Image object
    """
    try:
        # Load font with proper size
        if font_path:
            # Use larger font size for better quality (4x for crisp subscripts)
            render_font_size = font_size * 4
            font = ImageFont.truetype(font_path, render_font_size)
        else:
            render_font_size = font_size * 4
            font = ImageFont.load_default()
        
        # Get text bounding box with RAQM for proper Khmer shaping
        dummy_img = Image.new('RGBA', (1, 1), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(dummy_img)
        
        # Use RAQM - it should handle Khmer automatically
        try:
            bbox = draw.textbbox((0, 0), text, font=font, 
                                layout_engine=ImageFont.Layout.RAQM)
        except:
            bbox = draw.textbbox((0, 0), text, font=font)
        
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add generous padding for subscripts and superscripts
        padding_x = 20
        padding_y = 25  # Extra vertical padding for subscripts
        img_width = max(text_width + padding_x * 2, 80)
        img_height = max(text_height + padding_y * 2, 60)
        
        # Create high-res image with extra space
        img = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw text with RAQM
        try:
            draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, 
                     font=font, fill=(0, 0, 0, 255), 
                     layout_engine=ImageFont.Layout.RAQM)
        except:
            # Fallback without RAQM
            draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, 
                     font=font, fill=(0, 0, 0, 255))
        
        return img
    except Exception as e:
        print(f"Error rendering text '{text}': {e}")
        import traceback
        traceback.print_exc()
        # Fallback: create minimal image
        img = Image.new('RGBA', (120, 50), color=(255, 255, 255, 255))
        return img


def generate_pdf(word_list_path, output_pdf, num_pages=1, num_words=10, font_size=6,
                 id_font_size=18, line_spacing=2.0, id_spacing=0.3,
                 box_width_px=960, box_height_px=384, target_dpi=300, seed=None):
    """
    Generate a PDF with random words from word list and handwriting boxes in 2-column layout.
    
    Args:
        word_list_path: Path to CSV file with id,word columns
        output_pdf: Path to save generated PDF
        num_pages: Number of pages to generate
        num_words: Number of random words per page
        font_size: Font size for Khmer text
        id_font_size: Font size for ID numbers
        line_spacing: Vertical spacing between entries (in inches)
        id_spacing: Horizontal spacing between ID and word (in inches)
        box_width_px: Handwriting box width in target pixels
        box_height_px: Handwriting box height in target pixels
        target_dpi: DPI used to convert target pixels to PDF points
        seed: Random seed for reproducibility
    
    Returns:
        list of (id, word) tuples in the order they appear in PDF
    """
    print("\n" + "="*60)
    print("Checking system for Khmer text support...")
    print("="*60)
    
    # Check for RAQM support
    check_raqm_support()
    
    # Find Khmer font for rendering
    khmer_font_path = find_khmer_font()
    
    if not khmer_font_path:
        print("\n⚠ Proceeding without Khmer font - text will be garbled!")
        print("="*60 + "\n")
    else:
        print("="*60 + "\n")
    
    # generate random words to pdf
    if seed is not None:
        random.seed(seed)
    
    words = load_word_list(word_list_path)
    
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    
    # Two-column layout parameters
    left_margin = 0.5 * inch
    column_width = 3.5 * inch
    column_gap = 0.5 * inch
    y_start = height - 1 * inch
    
    # Box parameters: convert desired pixel dimensions to PDF points.
    # reportlab uses points (72 points = 1 inch).
    box_width = (box_width_px / target_dpi) * inch
    box_height = (box_height_px / target_dpi) * inch
    
    # Process words in 2-column layout (5 rows x 2 columns)
    rows_per_column = 5
    
    all_selected_words = []

    for page_number in range(1, num_pages + 1):
        page_words = sample_words(words, num_words)
        box_rects = []

        for idx, (word_id, word) in enumerate(page_words):
            # Determine column and row
            col = idx // rows_per_column  # 0 for left column, 1 for right column
            row = idx % rows_per_column   # 0-4 for each column

            # Calculate x position for this column
            x_position = left_margin + (col * (column_width + column_gap))

            # Calculate y position for this row
            y_position = y_start - (row * line_spacing * inch)

            # Draw ID (simple text, no complex rendering needed)
            id_text = f"{word_id}"
            c.setFont("Helvetica", id_font_size)
            c.drawString(x_position, y_position, id_text)

            # Calculate ID width and word position
            id_width = c.stringWidth(id_text, "Helvetica", id_font_size)
            word_x = x_position + id_width + (id_spacing * inch)

            # Render Khmer word as image for proper display
            word_img = render_text_as_image(word, font_size=font_size, font_path=khmer_font_path)

            # Convert PIL image to format ReportLab can use
            img_buffer = io.BytesIO()
            word_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)

            # Calculate image dimensions in PDF units (points)
            img_width_pts = word_img.width * 0.75  # Convert pixels to points (roughly)
            img_height_pts = word_img.height * 0.75

            # Draw image at calculated position
            # Adjust y position slightly to align with baseline
            img_y = y_position - (img_height_pts * 0.2)
            c.drawImage(img_reader, word_x, img_y,
                       width=img_width_pts, height=img_height_pts,
                       preserveAspectRatio=True, mask='auto')

            # Draw handwriting box below the text
            box_y = y_position - 0.25 * inch  # Small gap below text
            box_bottom = box_y - box_height

            box_rects.append((x_position, box_bottom, box_width, box_height))
            all_selected_words.append({
                'page': page_number,
                'position': idx + 1,
                'word_id': word_id,
                'word_text': word,
            })

        c.setStrokeColorRGB(0.3, 0.3, 0.3)  # Dark gray
        c.setLineWidth(1)
        for x, y, w, h in box_rects:
            c.rect(x, y, w, h, fill=0)

        if page_number < num_pages:
            c.showPage()
    
    c.save()
    
    print(f"Generated PDF: {output_pdf}")
    print(f"Pages: {num_pages}")
    print(f"Total words: {len(all_selected_words)}")
    
    return all_selected_words


def pdf_to_png(pdf_path, png_path, dpi=300):
    # convert pdf to png
    try:
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, dpi=dpi)
        
        if len(images) == 1:
            images[0].save(png_path, 'PNG')
            print(f"Converted to PNG: {png_path}")
        else:
            base_path = Path(png_path)
            stem = base_path.stem
            parent = base_path.parent
            
            for i, image in enumerate(images):
                page_path = parent / f"{stem}_page{i+1}.png"
                image.save(page_path, 'PNG')
                print(f"Converted page {i+1} to PNG: {page_path}")
        
        return True
    except Exception as e:
        print(f"error converting pdf to png: {e}")
        return False


def get_next_pdf_number(output_dir='pdfs_to_print'):
    """Find the next available PDF number in the output directory"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all existing pdf files
    existing_pdfs = list(output_path.glob('pdf*.pdf'))
    
    if not existing_pdfs:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for pdf_file in existing_pdfs:
        # Extract number from filename like 'pdf1.pdf', 'pdf2.pdf'
        match = pdf_file.stem.replace('pdf', '')
        if match.isdigit():
            numbers.append(int(match))
    
    if not numbers:
        return 1
    
    return max(numbers) + 1


def append_metadata(selected_words, pdf_filename, metadata_csv='pdfs_to_print/word_labels.csv'):
    """Append metadata to master CSV file"""
    metadata_path = Path(metadata_csv)
    metadata_path.parent.mkdir(exist_ok=True, parents=True)

    # Migrate legacy 4-column CSVs to the new page-aware schema.
    if metadata_path.exists():
        with open(metadata_path, 'r', newline='') as f:
            rows = list(csv.reader(f))

        if rows and rows[0] == ['pdf_file', 'word_id', 'word_text', 'position']:
            migrated_rows = [['pdf_file', 'page', 'word_id', 'word_text', 'position']]
            for row in rows[1:]:
                if not row:
                    continue
                if len(row) >= 5:
                    migrated_rows.append(row[:5])
                elif len(row) == 4:
                    migrated_rows.append([row[0], '1', row[1], row[2], row[3]])

            with open(metadata_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(migrated_rows)

    # Check if file exists to determine if we need to write header
    file_exists = metadata_path.exists()
    
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['pdf_file', 'page', 'word_id', 'word_text', 'position'])
        
        # Write data
        for item in selected_words:
            writer.writerow([
                pdf_filename,
                item['page'],
                item['word_id'],
                item['word_text'],
                item['position'],
            ])
    
    print(f"Appended metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PDFs with labeled words for handwriting dataset creation'
    )
    parser.add_argument('--word-list', '-w', 
                        default='dataset_generation/merged_corpus.csv',
                        help='Path to word list CSV (id,word columns)')
    parser.add_argument('--output-dir', '-d',
                        default='pdfs_to_print',
                        help='Output directory for PDFs (default: pdfs_to_print)')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Specific output PDF path (overrides auto-naming)')
    parser.add_argument('--num-pages', '-p', type=int, default=1,
                        help='Number of pages to generate in the PDF')
    parser.add_argument('--num-words', '-n', type=int, default=10,
                        help='Number of words to include per page')
    parser.add_argument('--font-size', type=int, default=6,
                        help='Font size for Khmer text')
    parser.add_argument('--id-font-size', type=int, default=18,
                        help='Font size for ID numbers')
    parser.add_argument('--box-width-px', type=int, default=960,
                        help='Handwriting box width in pixels at --dpi (default: 960)')
    parser.add_argument('--box-height-px', type=int, default=384,
                        help='Handwriting box height in pixels at --dpi (default: 384)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--convert-png', action='store_true',
                        help='Also convert PDF to PNG')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for PNG conversion')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        # Use specific output path if provided
        output_pdf = args.output
        Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
        pdf_filename = Path(output_pdf).name
    else:
        # Auto-generate filename in pdfs_to_print directory
        pdf_num = get_next_pdf_number(args.output_dir)
        pdf_filename = f'pdf{pdf_num}.pdf'
        output_pdf = str(Path(args.output_dir) / pdf_filename)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    selected_words = generate_pdf(
        args.word_list,
        output_pdf,
        num_pages=args.num_pages,
        num_words=args.num_words,
        font_size=args.font_size,
        id_font_size=args.id_font_size,
        box_width_px=args.box_width_px,
        box_height_px=args.box_height_px,
        target_dpi=args.dpi,
        seed=args.seed
    )
    
    # Append to master metadata CSV
    metadata_csv = str(Path(args.output_dir) / 'word_labels.csv')
    append_metadata(selected_words, pdf_filename, metadata_csv)
    
    if args.convert_png:
        png_path = Path(output_pdf).with_suffix('.png')
        pdf_to_png(output_pdf, png_path, dpi=args.dpi)


if __name__ == '__main__':
    main()

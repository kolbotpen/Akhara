"""
Convert PDF to PNG
Simple script to convert PDF files to PNG images for extraction
"""
import argparse
from pdf2image import convert_from_path
from pathlib import Path


def pdf_to_png(pdf_path, output_path=None, dpi=300):
    """
    Convert PDF to PNG image
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path for output PNG (optional, auto-generated if not provided)
        dpi: DPI resolution for conversion (default: 300)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = pdf_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {pdf_path} to PNG...")
    print(f"DPI: {dpi}")
    
    # Convert PDF to images
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    
    # Save first page as PNG
    pages[0].save(str(output_path), 'PNG')
    
    print(f"✓ Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF to PNG for handwriting extraction'
    )
    parser.add_argument('pdf', help='Path to input PDF file')
    parser.add_argument('--output', '-o', 
                        help='Output PNG path (default: same name as PDF with .png extension)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI resolution (default: 300)')
    
    args = parser.parse_args()
    
    try:
        pdf_to_png(args.pdf, args.output, args.dpi)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

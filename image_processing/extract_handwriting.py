"""
Extract handwriting from filled PDFs
Extracts handwriting boxes and labels them based on word_labels.csv
"""
import cv2
import numpy as np
import os
import csv
import argparse
from pathlib import Path
from pdf2image import convert_from_path


def load_labels_for_pdf(pdf_filename, labels_csv='pdfs_to_print/word_labels.csv'):
    """Load word labels for a specific PDF from the master CSV"""
    labels = []
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['pdf_file'] == pdf_filename:
                labels.append({
                    'word_id': int(row['word_id']),
                    'word_text': row['word_text'],
                    'position': int(row['position'])
                })
    
    # Sort by position to maintain order
    labels.sort(key=lambda x: x['position'])
    return labels


def pdf_to_image(pdf_path, dpi=300):
    """Convert PDF to image"""
    images = convert_from_path(pdf_path, dpi=dpi)
    if len(images) > 0:
        # Convert PIL image to OpenCV format
        img_array = np.array(images[0])
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    return None


def extract_handwriting_boxes(image, num_boxes=10, columns=2, rows=5):
    """
    Extract handwriting boxes from the image based on expected layout.
    
    Args:
        image: OpenCV image (BGR)
        num_boxes: Total number of boxes expected
        columns: Number of columns in layout
        rows: Number of rows per column
    
    Returns:
        List of extracted box images in order (left column top to bottom, then right column)
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to find boxes
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find box-like rectangles
    box_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Boxes should be wide rectangles with reasonable size
        # Adjust these thresholds based on your PDF layout
        if area > 50000 and 3 < aspect_ratio < 8 and w > 400:
            box_contours.append((x, y, w, h))
    
    # Sort boxes: first by column (x position), then by row (y position)
    # Group by columns first
    if len(box_contours) >= num_boxes:
        # Sort by y-position first to get rows, then by x to get columns
        box_contours.sort(key=lambda b: (b[0], b[1]))
        
        # Split into columns
        mid_x = width / 2
        left_column = sorted([b for b in box_contours if b[0] < mid_x], key=lambda b: b[1])
        right_column = sorted([b for b in box_contours if b[0] >= mid_x], key=lambda b: b[1])
        
        # Combine: left column first, then right column
        ordered_boxes = left_column[:rows] + right_column[:rows]
    else:
        # Fallback: just sort by position
        box_contours.sort(key=lambda b: (b[1], b[0]))
        ordered_boxes = box_contours[:num_boxes]
    
    # Extract box images
    extracted_boxes = []
    for x, y, w, h in ordered_boxes:
        # Add small padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        box_img = image[y1:y2, x1:x2]
        extracted_boxes.append(box_img)
    
    return extracted_boxes, ordered_boxes


def process_pdf(pdf_path, output_dir='extracted_handwriting', labels_csv='pdfs_to_print/word_labels.csv', 
                visualize=True, dpi=300):
    """
    Extract handwriting from a filled PDF and save with labels.
    
    Args:
        pdf_path: Path to the filled PDF
        output_dir: Directory to save extracted images
        labels_csv: Path to word_labels.csv
        visualize: Create visualization showing detected boxes
        dpi: Resolution for PDF conversion
    """
    pdf_path = Path(pdf_path)
    pdf_filename = pdf_path.name
    
    print(f"Processing: {pdf_filename}")
    
    # Load labels for this PDF
    labels = load_labels_for_pdf(pdf_filename, labels_csv)
    
    if not labels:
        print(f"Error: No labels found for {pdf_filename} in {labels_csv}")
        return
    
    print(f"Found {len(labels)} labels for this PDF")
    
    # Convert PDF to image
    print("Converting PDF to image...")
    image = pdf_to_image(pdf_path, dpi=dpi)
    
    if image is None:
        print("Error: Could not convert PDF to image")
        return
    
    # Extract handwriting boxes
    print("Extracting handwriting boxes...")
    extracted_boxes, box_coords = extract_handwriting_boxes(image, num_boxes=len(labels))
    
    if len(extracted_boxes) != len(labels):
        print(f"Warning: Found {len(extracted_boxes)} boxes but expected {len(labels)}")
        print("Proceeding with available boxes...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save extracted boxes with labels
    labeled_data = []
    for idx, (box_img, label) in enumerate(zip(extracted_boxes, labels)):
        word_id = label['word_id']
        word_text = label['word_text']
        
        # Generate filename
        filename = f"{pdf_path.stem}_{word_text}_{word_id:04d}_{idx:02d}.png"
        filepath = output_path / filename
        
        # Save image
        cv2.imwrite(str(filepath), box_img)
        
        labeled_data.append({
            'source_pdf': pdf_filename,
            'image_path': filename,
            'word_id': word_id,
            'word_text': word_text,
            'position': idx + 1
        })
        
        print(f"  Saved: {filename} (word: '{word_text}')")
    
    # Save metadata CSV
    metadata_path = output_path / f'{pdf_path.stem}_labels.csv'
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_pdf', 'image_path', 'word_id', 'word_text', 'position'])
        writer.writeheader()
        writer.writerows(labeled_data)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Create visualization
    if visualize and box_coords:
        viz_img = image.copy()
        for idx, (x, y, w, h) in enumerate(box_coords[:len(labels)]):
            # Draw box
            cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Add label
            if idx < len(labels):
                label_text = f"{idx+1}: {labels[idx]['word_text']}"
                cv2.putText(viz_img, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        viz_path = output_path / f'{pdf_path.stem}_visualization.png'
        cv2.imwrite(str(viz_path), viz_img)
        print(f"Visualization saved to: {viz_path}")
    
    print(f"\n✓ Successfully extracted {len(labeled_data)} handwriting samples")
    return labeled_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract handwriting from filled PDFs and create labeled dataset'
    )
    parser.add_argument('pdf', help='Path to filled PDF file')
    parser.add_argument('--output', '-o', 
                        default='extracted_handwriting',
                        help='Output directory (default: extracted_handwriting)')
    parser.add_argument('--labels-csv', '-l',
                        default='pdfs_to_print/word_labels.csv',
                        help='Path to word_labels.csv file')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip creating visualization')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for PDF conversion (default: 300)')
    
    args = parser.parse_args()
    
    try:
        process_pdf(
            args.pdf,
            output_dir=args.output,
            labels_csv=args.labels_csv,
            visualize=not args.no_viz,
            dpi=args.dpi
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

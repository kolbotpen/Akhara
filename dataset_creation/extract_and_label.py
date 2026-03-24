"""
Extract and Label Boxes from Generated PDFs
Extracts text boxes, reads IDs via OCR, and creates labeled dataset
"""
import cv2
import numpy as np
import os
import csv
import argparse
from pathlib import Path
import pytesseract
import re


def load_word_mapping(word_list_path):
    """Load word list from CSV file"""
    words = {}
    with open(word_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            words[int(row['id'])] = row['word']
    return words


def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def remove_overlapping_boxes(boxes, iou_threshold=0.5):
    """Remove duplicate/overlapping boxes"""
    if not boxes:
        return boxes
    
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    
    keep = []
    for box in boxes:
        overlaps = False
        for kept_box in keep:
            if calculate_iou(box, kept_box) > iou_threshold:
                overlaps = True
                break
        
        if not overlaps:
            keep.append(box)
    
    return keep


def extract_text_boxes(image, min_area=1000, min_aspect_ratio=2.0):
    """
    Extract text boxes from image.
    
    Args:
        image: OpenCV image (BGR)
        min_area: Minimum box area to consider
        min_aspect_ratio: Minimum width/height ratio
    
    Returns:
        List of boxes (x, y, w, h) sorted top-to-bottom
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter: must be large enough and wide enough
        if area > min_area and aspect_ratio > min_aspect_ratio:
            boxes.append((x, y, w, h))
    
    # Remove overlapping boxes
    boxes = remove_overlapping_boxes(boxes)
    
    # Sort top to bottom, left to right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    return boxes


def extract_id_from_box(image, box, debug=False, debug_name="debug"):
    """
    Extract ID from ABOVE a handwriting box using OCR.
    The ID is printed above the box, not inside it.
    
    Args:
        image: Full image
        box: (x, y, w, h) of the handwriting box
        debug: If True, save debug images
        debug_name: Prefix for debug image files
    
    Returns:
        int: Extracted ID, or None if not found
    """
    x, y, w, h = box
    
    #Look ABOVE the box for the ID text
    # Estimate text line height (about 0.3-0.5 inches at 300 DPI = 90-150 pixels)
    # Increase search area for boxes that might be at the top
    text_height = 200  # pixels (increased further for problematic boxes)
    search_y_start = max(0, y - text_height)
    search_y_end = y  # Just above the box
    
    # Extract left portion where ID should be (IDs are 1-2 digits, so narrow region)
    id_width = min(150, int(w * 0.25))  # Narrower - just the ID number
    id_region = image[search_y_start:search_y_end, x:x+id_width]
    
    if debug:
        cv2.imwrite(f'{debug_name}_id_region.png', id_region)
    
    # Check if region is mostly blank (might need to expand search area)
    if id_region.size > 0:
        gray_check = cv2.cvtColor(id_region, cv2.COLOR_BGR2GRAY)
        non_white_pixels = np.count_nonzero(gray_check < 240)
        total_pixels = gray_check.size
        if non_white_pixels < total_pixels * 0.01:  # Less than 1% non-white
            if debug:
                print(f"    Region mostly blank, expanding search...")
            # Try expanding search area upward
            text_height_expanded = 250
            search_y_start_expanded = max(0, y - text_height_expanded)
            id_region = image[search_y_start_expanded:search_y_end, x:x+id_width]
    
    # Preprocess for better OCR - use adaptive thresholding for small text
    gray = cv2.cvtColor(id_region, cv2.COLOR_BGR2GRAY)
    
    # Try adaptive thresholding which works better for small text
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Invert if most pixels are black (text should be black on white for OCR)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    
    # Scale up significantly for better OCR on small numbers
    scale = 5  # Increased from 3 to 5 for tiny text
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if debug:
        cv2.imwrite(f'{debug_name}_id_processed.png', gray)
    
    # Use pytesseract to extract text
    # Configure to recognize only digits
    # Try multiple PSM modes for better detection of small numbers
    configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789', # Raw line
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',  # Uniform block
    ]
    
    for config in configs:
        text = pytesseract.image_to_string(gray, config=config)
        if debug:
            print(f"    OCR attempt with config '{config.split('--psm')[1].split()[0]}': '{text.strip()}'")
        
        # Extract first number found
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
    
    if debug:
        print(f"    All OCR attempts failed")
    
    return None


def extract_word_from_box(image, box, id_width_ratio=0.3):
    """
    Extract the handwriting from inside the box.
    
    Args:
        image: Full image
        box: (x, y, w, h) of the handwriting box
        id_width_ratio: Not used (kept for compatibility)
    
    Returns:
        numpy array: Image of the handwriting inside the box
    """
    x, y, w, h = box
    
    # Extract the full box content (the handwriting)
    # Add small padding to avoid box borders
    padding = 5
    x_start = max(0, x + padding)
    y_start = max(0, y + padding)
    x_end = min(image.shape[1], x + w - padding)
    y_end = min(image.shape[0], y + h - padding)
    
    handwriting = image[y_start:y_end, x_start:x_end]
    
    return handwriting


def process_image_and_label(image_path, word_list_path, output_dir='labeled_data',
                            min_area=1000, min_aspect_ratio=2.0, debug=False):
    """
    Main function to extract handwriting boxes, read IDs from above boxes, and create labeled dataset.
    
    Process:
    1. Find handwriting boxes (empty rectangles in the PDF)
    2. Look above each box to find the ID number
    3. Extract the handwriting from inside each box
    4. Label with corresponding word from word list
    
    Args:
        image_path: Path to input image (PNG from filled PDF)
        word_list_path: Path to word list CSV
        output_dir: Directory to save labeled images and CSV
        min_area: Minimum box area threshold
        min_aspect_ratio: Minimum width/height ratio
        debug: Enable debug mode
    
    Returns:
        List of (image_path, word_id, word_text) tuples
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Load word mapping
    word_mapping = load_word_mapping(word_list_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Extract text boxes
    print(f"Extracting boxes from {image_path}...")
    boxes = extract_text_boxes(image, min_area, min_aspect_ratio)
    print(f"Found {len(boxes)} text boxes")
    
    # Process each box
    labeled_data = []
    successful = 0
    failed = 0
    word_counts = {}  # Track how many times each word appears
    
    for idx, box in enumerate(boxes):
        # Extract ID from above the box (enable debug for all boxes if debug mode is on)
        word_id = extract_id_from_box(image, box, debug=debug, debug_name=f"box{idx+1}")
        
        if word_id is None:
            print(f"  Box {idx+1}: Could not extract ID (looked above box), skipping")
            if debug:
                x, y, w, h = box
                print(f"    Box position: x={x}, y={y}, w={w}, h={h}")
            failed += 1
            continue
        
        # Get word from mapping
        if word_id not in word_mapping:
            print(f"  Box {idx+1}: ID {word_id} not in word list, skipping")
            failed += 1
            continue
        
        word_text = word_mapping[word_id]
        
        # Extract handwriting from inside the box
        handwriting_image = extract_word_from_box(image, box)
        
        # Track word occurrences and generate filename
        word_counts[word_text] = word_counts.get(word_text, 0) + 1
        count = word_counts[word_text]
        
        # Save handwriting image with word label as filename
        image_filename = f'{word_text}_{count}.png'
        image_save_path = images_dir / image_filename
        cv2.imwrite(str(image_save_path), handwriting_image)
        
        # Relative path for CSV
        relative_path = f'images/{image_filename}'
        labeled_data.append((relative_path, word_id, word_text))
        
        print(f"  Box {idx+1}: ID={word_id}, Word='{word_text}' -> {image_filename}")
        successful += 1
    
    print(f"\nSuccessfully labeled: {successful}")
    print(f"Failed: {failed}")
    
    # Save labeled data to CSV
    csv_path = output_path / 'labels.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'word_id', 'word_text'])
        writer.writerows(labeled_data)
    
    print(f"\nLabeled data saved to: {csv_path}")
    print(f"Images saved to: {images_dir}")
    
    # Create visualization
    viz_image = image.copy()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    viz_path = output_path / 'visualization.png'
    cv2.imwrite(str(viz_path), viz_image)
    print(f"Visualization saved to: {viz_path}")
    
    return labeled_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract and label text boxes from generated PDFs'
    )
    parser.add_argument('image', help='Path to input PNG image')
    parser.add_argument('--word-list', '-w',
                        default='data/word_list.csv',
                        help='Path to word list CSV')
    parser.add_argument('--output', '-o',
                        default='labeled_data',
                        help='Output directory for labeled data')
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Minimum box area threshold')
    parser.add_argument('--min-aspect-ratio', type=float, default=2.0,
                        help='Minimum width/height ratio')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        labeled_data = process_image_and_label(
            args.image,
            args.word_list,
            args.output,
            args.min_area,
            args.min_aspect_ratio,
            args.debug
        )
        print(f"\n✓ Successfully processed {len(labeled_data)} labeled samples")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

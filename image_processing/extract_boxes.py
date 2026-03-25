import cv2
import numpy as np
import os
from pathlib import Path

def calculate_iou(box1, box2):
    # Calculate Intersection over Union of the two boxes
    
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
    # Remove duplicate/overlapping boxes
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

def extract_text_boxes(image_path, output_dir='extracted_boxes', min_area=1000, min_aspect_ratio=3.0):

    Path(output_dir).mkdir(exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter: must be large enough and wide enough (boxes are wider than tall)
        if area > min_area and aspect_ratio > min_aspect_ratio:
            boxes.append((x, y, w, h))
    
    boxes = remove_overlapping_boxes(boxes)
    
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    print(f"Found {len(boxes)} text boxes")
    
    for idx, (x, y, w, h) in enumerate(boxes):
        box_img = img[y:y+h, x:x+w]
        
        output_path = os.path.join(output_dir, f'box_{idx:03d}.png')
        cv2.imwrite(output_path, box_img)
        print(f"Saved: {output_path} ({w}x{h})")
    
    # Create a visualization showing detected boxes
    viz_img = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    viz_path = os.path.join(output_dir, 'detected_boxes_visualization.png')
    cv2.imwrite(viz_path, viz_img)
    print(f"\nVisualization saved to: {viz_path}")
    
    return boxes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text boxes from an image')
    parser.add_argument('image', help='Path to the input PNG image')
    parser.add_argument('--output', '-o', default='extracted_boxes', 
                        help='Output directory for extracted boxes (default: extracted_boxes)')
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Minimum area threshold to filter out noise (default: 1000)')
    parser.add_argument('--min-aspect-ratio', type=float, default=3.0,
                        help='Minimum width/height ratio to filter boxes (default: 3.0)')
    
    args = parser.parse_args()
    
    try:
        boxes = extract_text_boxes(args.image, args.output, args.min_area, args.min_aspect_ratio)
        print(f"\nSuccessfully extracted {len(boxes)} boxes to '{args.output}/'")
    except Exception as e:
        print(f"Error: {e}")

"""
Test script for grid corner detection.
Outputs visualization to check if corners are detected correctly.
"""

import sys
import cv2
import numpy as np
from parking_grid_homography import find_grid_corners, visualize_corner_detection


def test_corner_detection(image_path: str, output_path: str, 
                         white_threshold: int = 200,
                         line_threshold: int = 50,
                         scale: float = 0.5):
    """Test corner detection on a single image."""
    print(f"Testing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return False
    
    print(f"  Image shape: {image.shape}")
    print(f"  Processing at scale {scale}x...")
    
    # Detect corners
    corners = find_grid_corners(image, white_threshold, line_threshold, scale=scale)
    
    if corners is None:
        print("  ❌ No corners detected")
        vis = image.copy()
    else:
        print(f"  ✓ Detected {len(corners)} corners:")
        for i, (x, y) in enumerate(corners):
            print(f"    [{i}] ({x:.1f}, {y:.1f})")
        vis = visualize_corner_detection(image, corners)
    
    cv2.imwrite(output_path, vis)
    print(f"  Saved visualization: {output_path}")
    return corners is not None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_corner_detection.py <image_path> [white_threshold] [line_threshold] [scale]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    white_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    line_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    # Output to same directory
    import os
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_corners_detected{ext}"
    
    success = test_corner_detection(image_path, output_path, white_threshold, line_threshold, scale)
    
    if not success:
        print("\n⚠️  Corners not detected. Try adjusting thresholds:")
        print("  - Increase white_threshold (100-255) if white lines are too dark")
        print("  - Decrease line_threshold (10-100) if Hough is too strict")
        print("  - Decrease scale (0.1-0.5) for faster processing")

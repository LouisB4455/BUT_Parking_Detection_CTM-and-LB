"""
Interactive corner marking tool for parking grid calibration.

Mark the 4 corners (top-left, top-right, bottom-right, bottom-left) on a reference image,
and the tool will save the homography transformation to align future images.
"""

import cv2
import numpy as np
import pickle
import sys
import os
from typing import List, Tuple, Optional


class CornerMarker:
    def __init__(self, image_path: str, max_display_width: int = 1000):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.image_path = image_path
        self.corners: List[Tuple[int, int]] = []
        
        # Compute display scale to fit screen
        h, w = self.image.shape[:2]
        if w > max_display_width:
            self.scale = max_display_width / w
        else:
            self.scale = 1.0
        
        # Create display image at scaled size
        disp_w = int(w * self.scale)
        disp_h = int(h * self.scale)
        self.display = cv2.resize(self.image, (disp_w, disp_h))
        
        print(f"Loaded image: {image_path}")
        print(f"Original size: {w}x{h}")
        print(f"Display size: {disp_w}x{disp_h} (scale: {self.scale:.2f})")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark corners."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                # Convert display coordinates to original image coordinates
                orig_x = int(x / self.scale)
                orig_y = int(y / self.scale)
                self.corners.append((orig_x, orig_y))
                print(f"Corner {len(self.corners)}: ({orig_x}, {orig_y}) [display: ({x}, {y})]")
                
                # Redraw display
                self.display = cv2.resize(self.image, (int(self.image.shape[1] * self.scale), 
                                                       int(self.image.shape[0] * self.scale)))
                
                for i, (orig_cx, orig_cy) in enumerate(self.corners):
                    # Convert back to display coordinates for drawing
                    disp_cx = int(orig_cx * self.scale)
                    disp_cy = int(orig_cy * self.scale)
                    cv2.circle(self.display, (disp_cx, disp_cy), 10, (0, 255, 0), 2)
                    label = ["TL", "TR", "BR", "BL"][i]
                    cv2.putText(self.display, label, (disp_cx + 20, disp_cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw bounding box if we have at least 2 corners
                if len(self.corners) >= 2:
                    pts = np.array([(int(c[0] * self.scale), int(c[1] * self.scale)) for c in self.corners], dtype=np.int32)
                    cv2.polylines(self.display, [pts], False, (0, 255, 0), 2)
                
                cv2.imshow("Mark Corners (Click to mark, SPACE when done)", self.display)
            else:
                print("All 4 corners marked. Press SPACE to finish.")
    
    def run(self) -> Optional[np.ndarray]:
        """
        Interactive corner marking.
        Returns 4x2 array of corners (top-left, top-right, bottom-right, bottom-left)
        """
        cv2.namedWindow("Mark Corners (Click to mark, SPACE when done)")
        cv2.setMouseCallback("Mark Corners (Click to mark, SPACE when done)", 
                            self.mouse_callback)
        cv2.imshow("Mark Corners (Click to mark, SPACE when done)", self.display)
        
        print("\nInstructions:")
        print("1. Click 4 corners in order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT")
        print("2. Press SPACE when done")
        print("3. Press ESC to cancel")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):  # SPACE
                if len(self.corners) == 4:
                    cv2.destroyAllWindows()
                    return np.array(self.corners, dtype=np.float32)
                else:
                    print(f"Need 4 corners, got {len(self.corners)}. Continue clicking...")
            elif key == 27:  # ESC
                print("Cancelled.")
                cv2.destroyAllWindows()
                return None
    
    def undo_last(self):
        """Remove the last corner."""
        if self.corners:
            self.corners.pop()
            print(f"Undo. Corners: {len(self.corners)}/4")
            self.display = cv2.resize(self.image, (int(self.image.shape[1] * self.scale), 
                                                   int(self.image.shape[0] * self.scale)))
            for i, (orig_cx, orig_cy) in enumerate(self.corners):
                disp_cx = int(orig_cx * self.scale)
                disp_cy = int(orig_cy * self.scale)
                cv2.circle(self.display, (disp_cx, disp_cy), 10, (0, 255, 0), 2)
                label = ["TL", "TR", "BR", "BL"][i]
                cv2.putText(self.display, label, (disp_cx + 20, disp_cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Mark Corners (Click to mark, SPACE when done)", self.display)


def compute_grid_bounds(image_shape: tuple) -> np.ndarray:
    """
    Compute expected corners for a "perfect" grid aligned with image bounds.
    Returns 4x2 array.
    """
    h, w = image_shape[:2]
    margin = 50  # Small margin from edges
    
    return np.array([
        [margin, margin],           # top-left
        [w - margin, margin],       # top-right
        [w - margin, h - margin],   # bottom-right
        [margin, h - margin],       # bottom-left
    ], dtype=np.float32)


def save_homography(src_corners: np.ndarray,
                   image_shape: tuple,
                   output_path: str):
    """
    Compute and save homography matrix for perspective correction.
    """
    # image_shape is (h, w, c), we need just (h, w)
    h, w = image_shape[:2]
    dst_corners = compute_grid_bounds(image_shape)
    
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Save to pickle
    data = {
        'H': H,
        'src_corners': src_corners,
        'dst_corners': dst_corners,
        'image_shape': image_shape,
        'description': 'Parking grid homography transformation'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Homography saved to: {output_path}")
    print(f"  Source corners shape: {src_corners.shape}")
    print(f"  Destination corners shape: {dst_corners.shape}")


def load_homography(path: str) -> Optional[np.ndarray]:
    """Load homography matrix from file."""
    if not os.path.exists(path):
        return None
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data.get('H')


def apply_homography(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply perspective transformation to image."""
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, (w, h))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate_grid_corners.py <reference_image> [output_homography_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "grid_homography.pkl"
    
    # Run corner marking
    try:
        marker = CornerMarker(image_path)
        corners = marker.run()
        
        if corners is not None:
            # Load image to get shape
            img = cv2.imread(image_path)
            save_homography(corners, img.shape, output_path)
            print(f"\n✓ Calibration complete!")
        else:
            print("Calibration cancelled.")
            sys.exit(1)
    
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

"""
Automatic grid corner detection and homography computation for parking zones.

Detects parking grid corners (white lines intersections) and computes perspective 
transformation to align zones across angle changes.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


def detect_white_lines(image: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Detect white lines in the image using HSV.
    Returns binary mask of white pixels.
    """
    # Convert to HSV for better white detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # White in HSV: high V (brightness), low S (saturation)
    lower_white = np.array([0, 0, threshold], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def detect_hough_lines(binary_mask: np.ndarray, 
                       rho: float = 1.0,
                       theta: float = np.pi / 180,
                       threshold: int = 50) -> List[Tuple[float, float, float]]:
    """
    Detect lines using Hough line transform.
    Returns list of (rho, theta, score).
    """
    lines = cv2.HoughLines(binary_mask, rho, theta, threshold)
    
    if lines is None:
        return []
    
    # Convert to list of tuples
    result = []
    for line in lines:
        rho, theta = line[0]
        result.append((rho, theta))
    
    return result


def line_equation(rho: float, theta: float) -> Tuple[float, float, float]:
    """
    Convert Hough parameters (rho, theta) to line equation ax + by + c = 0.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


def line_intersection(line1: Tuple[float, float, float], 
                     line2: Tuple[float, float, float],
                     image_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """
    Find intersection of two lines given as (a, b, c) where ax + by + c = 0.
    Returns (x, y) if intersection is within image bounds, else None.
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None  # Lines are parallel
    
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    
    h, w = image_shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return x, y
    
    return None


def find_grid_corners(image: np.ndarray,
                     white_threshold: int = 200,
                     line_threshold: int = 50,
                     scale: float = 0.5) -> Optional[np.ndarray]:
    """
    Detect parking grid corners from white lines.
    
    Args:
        image: Input image
        white_threshold: Brightness threshold for white detection
        line_threshold: Hough line threshold
        scale: Scale factor for processing (smaller = faster)
    
    Returns:
        4x2 array of corner coordinates (x, y) in order: top-left, top-right, bottom-right, bottom-left
        Or None if detection fails.
    """
    # Scale down for faster processing
    if scale < 1.0:
        h, w = image.shape[:2]
        small_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        small_image = image
    
    # Detect white lines on small image
    white_mask = detect_white_lines(small_image, white_threshold)
    
    # Extract lines
    hough_lines = detect_hough_lines(white_mask, threshold=line_threshold)
    
    if len(hough_lines) < 4:
        # Not enough lines detected
        return None
    
    # Convert to line equations
    line_eqs = [line_equation(rho, theta) for rho, theta in hough_lines]
    
    # Find intersections
    intersections = []
    for i in range(len(line_eqs)):
        for j in range(i + 1, len(line_eqs)):
            pt = line_intersection(line_eqs[i], line_eqs[j], small_image.shape)
            if pt is not None:
                intersections.append(pt)
    
    if len(intersections) < 4:
        return None
    
    # Cluster nearby intersections (they might be detected multiple times)
    intersections = cluster_points(intersections, distance_threshold=20)
    
    if len(intersections) < 4:
        return None
    
    # Scale back to original image size
    if scale < 1.0:
        intersections = [(x / scale, y / scale) for x, y in intersections]
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    # Sort by y first, then x
    intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
    
    # Top two points
    top_points = sorted(intersections[:2], key=lambda p: p[0])
    # Bottom two points
    bottom_points = sorted(intersections[2:4], key=lambda p: p[0])
    
    corners = np.array([
        top_points[0],      # top-left
        top_points[1],      # top-right
        bottom_points[1],   # bottom-right
        bottom_points[0],   # bottom-left
    ], dtype=np.float32)
    
    return corners


def cluster_points(points: List[Tuple[float, float]], 
                   distance_threshold: float = 20) -> List[Tuple[float, float]]:
    """
    Cluster nearby points and return their centroids.
    """
    if not points:
        return []
    
    points = np.array(points, dtype=np.float32)
    
    # Use DBSCAN-like clustering
    visited = [False] * len(points)
    clusters = []
    
    for i in range(len(points)):
        if visited[i]:
            continue
        
        cluster = [points[i]]
        visited[i] = True
        
        for j in range(i + 1, len(points)):
            if visited[j]:
                continue
            
            dist = np.linalg.norm(points[i] - points[j])
            if dist < distance_threshold:
                cluster.append(points[j])
                visited[j] = True
        
        centroid = np.mean(cluster, axis=0)
        clusters.append(tuple(centroid))
    
    return clusters


def compute_homography(src_corners: np.ndarray,
                      dst_corners: np.ndarray) -> np.ndarray:
    """
    Compute perspective transformation matrix (homography).
    
    Args:
        src_corners: 4x2 array of source corners
        dst_corners: 4x2 array of destination corners
    
    Returns:
        3x3 homography matrix
    """
    H, _ = cv2.getPerspectiveTransform(src_corners, dst_corners)
    return H


def apply_homography_to_polygon(polygon: List[Tuple[float, float]],
                               H: np.ndarray) -> List[Tuple[float, float]]:
    """
    Apply homography transformation to a polygon.
    
    Args:
        polygon: List of (x, y) coordinates
        H: 3x3 homography matrix
    
    Returns:
        Transformed polygon as list of (x, y) coordinates
    """
    points = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H)
    return [tuple(pt[0]) for pt in transformed]


def visualize_corner_detection(image: np.ndarray,
                              corners: Optional[np.ndarray],
                              save_path: Optional[str] = None) -> np.ndarray:
    """
    Draw detected corners on the image for visualization.
    """
    vis = image.copy()
    
    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv2.putText(vis, f"{i}", (int(x) + 15, int(y) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw bounding box
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis)
    
    return vis

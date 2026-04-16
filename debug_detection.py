#!/usr/bin/env python3
"""Quick debug script to test car detection on sample image."""

import sys
import cv2
import numpy as np
from pathlib import Path
import pickle

sys.path.insert(0, r"c:\Users\cdeth\OneDrive - ISEP\Documents\Projet ISEP VUT\BUT_Parking_Detection_CTM-and-LB\modèle final")

from ultralytics import YOLO

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("modèle final/yolov8m.pt")

# Test image
test_image = r"DATA/2024-04-05/2024-04-05 17_04_26.jpg"
print(f"Loading image: {test_image}")
frame = cv2.imread(test_image)
if frame is None:
    print("ERROR: Could not load image!")
    sys.exit(1)

print(f"Image shape: {frame.shape}")

# Load reference line
print("Loading reference line...")
with open("modèle final/ligne_reference.pkl", "rb") as f:
    ref_data = pickle.load(f)
line_ref = ref_data['line']
print(f"Reference line: {line_ref}")

# YOLO detection
print("\nRunning YOLO detection...")
results = model(frame, conf=0.5, verbose=False)

cars = []
car_centers = []
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    print(f"  Found {len(boxes)} detections total")
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        print(f"    Detection {i}: class={int(cls)}, box={box}")
        if int(cls) != 2:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        car_centers.append((cx, cy))
        cars.append({'bbox': (x1, y1, x2, y2), 'center': (cx, cy)})

print(f"\nCars detected: {len(cars)}")
if car_centers:
    for i, (cx, cy) in enumerate(car_centers):
        print(f"  Car {i}: center=({cx}, {cy})")

# Try to fit line
if len(car_centers) >= 2:
    print("\nFitting line to car centers...")
    points = np.array(car_centers, dtype=np.float32)
    try:
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line
        print(f"  Fitted line: vx={vx[0]}, vy={vy[0]}, x0={x0[0]}, y0={y0[0]}")
        
        # Normalize direction vector
        norm = np.sqrt(vx[0]**2 + vy[0]**2)
        vx = vx[0] / norm
        vy = vy[0] / norm
        
        # Get two points on the line
        t1 = -500
        t2 = 500
        p1 = (x0[0] + t1 * vx, y0[0] + t1 * vy)
        p2 = (x0[0] + t2 * vx, y0[0] + t2 * vy)
        print(f"  Line endpoints: {p1} to {p2}")
        
        # Check which endpoints are visible on image
        h, w = frame.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Clip to image bounds
        def clip_point(p):
            x, y = p
            return (max(0, min(w-1, x)), max(0, min(h-1, y)))
        p1_clipped = clip_point(p1)
        p2_clipped = clip_point(p2)
        print(f"  Clipped endpoints: {p1_clipped} to {p2_clipped}")
        
        # Compute midpoints
        ref_mid = ((line_ref[0][0] + line_ref[1][0]) / 2.0, (line_ref[0][1] + line_ref[1][1]) / 2.0)
        cur_mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        
        print(f"\n  Reference line midpoint: {ref_mid}")
        print(f"  Current line midpoint: {cur_mid}")
        
        dx = cur_mid[0] - ref_mid[0]
        dy = cur_mid[1] - ref_mid[1]
        print(f"  Delta: dx={dx:.1f}, dy={dy:.1f}")
        
    except Exception as e:
        print(f"  ERROR fitting line: {e}")
else:
    print(f"Not enough cars to fit line (need >= 2, got {len(car_centers)})")

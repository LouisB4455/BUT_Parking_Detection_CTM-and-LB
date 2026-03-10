import cv2
import pickle
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================

IMAGE_PATH = "photo/2026-03-09_1739.jpg"
PICKLE_PATH = "parking_slots.pkl"
MODEL_PATH = "yolov8m.pt"

# =========================
# Chargement modèle YOLO
# =========================

model = YOLO(MODEL_PATH)

# =========================
# Chargement places parking
# =========================

with open(PICKLE_PATH, "rb") as f:
    pos_list = pickle.load(f)

# =========================
# Resize pour écran
# =========================

def resize_to_screen(image, max_width=1400, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

# =========================
# Chargement image
# =========================

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Erreur lecture image")
    exit()

print("Analyse :", IMAGE_PATH)

# =========================
# Détection YOLO (COCO standard)
# =========================

results = model(frame, conf=0.35)

detections = []

for r in results:

    boxes = r.boxes.xyxy.cpu().numpy()   # x1, y1, x2, y2
    classes = r.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):

        cls = int(cls)
        if cls == 2:  # COCO car
            x1, y1, x2, y2 = box.astype(int)
            pts = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])
            detections.append(pts)
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

# =========================
# Vérification places parking
# =========================

free_places = 0

for poly in pos_list:

    parking_poly = np.array(poly, np.int32)
    occupied = False

    for det in detections:

        intersection_area = cv2.intersectConvexConvex(
            parking_poly.astype(np.float32),
            det.astype(np.float32)
        )[0]

        parking_area = cv2.contourArea(parking_poly)
        occupied = (intersection_area / parking_area) > 0.3

        if occupied:
            break

    if occupied:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
        free_places += 1

    cv2.polylines(frame, [parking_poly], True, color, 3)

# =========================
# Compteur places libres
# =========================

total = len(pos_list)
cv2.rectangle(frame, (20,20), (350,80), (0,0,0), -1)
cv2.putText(
    frame,
    f"Free: {free_places} / {total}",
    (40,65),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0,255,0),
    3
)

# =========================
# Affichage
# =========================

display = resize_to_screen(frame)

cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Parking Detection", display)

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

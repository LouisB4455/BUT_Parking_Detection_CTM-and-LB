import cv2
import pickle
import numpy as np
import os
import glob
import csv
from ultralytics import YOLO

# CONFIGURATION
FOLDER_PATH = "photo/"
PICKLE_PATH = "parking_slots.pkl"
MODEL_PATH = "yolov8m.pt"
CSV_PATH = "results_analyse_1.csv"

# Chargement modèle YOLO
model = YOLO(MODEL_PATH)

# Chargement places parking
with open(PICKLE_PATH, "rb") as f:
    pos_list = pickle.load(f)

# Lister toutes les images
image_extensions = ['*.jpg', '*.jpeg', '*.png']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))

if not image_files:
    print(f"Aucune image trouvée dans le dossier {FOLDER_PATH}")
    exit()

print(f"Début du traitement : {len(image_files)} images trouvées.")

# Création du CSV + header
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "free_places", "occupied_places", "total_places"])

    # BOUCLE DE TRAITEMENT
    for img_path in image_files:
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Saut de l'image (erreur lecture) : {img_path}")
            continue

        file_name = os.path.basename(img_path)
        print(f"Analyse de {file_name}...")

        # Détection YOLO
        results = model(frame, conf=0.35, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 2:  # car
                    x1, y1, x2, y2 = box.astype(int)
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    detections.append(pts)
                    cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

        # Vérification places parking
        free_places = 0
        occupied_places = 0

        for poly in pos_list:
            parking_poly = np.array(poly, np.int32)
            occupied = False

            for det in detections:
                res, _ = cv2.intersectConvexConvex(
                    parking_poly.astype(np.float32),
                    det.astype(np.float32)
                )

                parking_area = cv2.contourArea(parking_poly)

                if parking_area > 0 and (res / parking_area) > 0.3:
                    occupied = True
                    break

            if occupied:
                occupied_places += 1
                color = (0, 0, 255)
            else:
                free_places += 1
                color = (0, 255, 0)

            cv2.polylines(frame, [parking_poly], True, color, 3)

        total = len(pos_list)

        # Écriture CSV
        writer.writerow([file_name, free_places, occupied_places, total])

        # Cartouche
        cv2.rectangle(frame, (20, 20), (650, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"Free: {free_places} / {total}", (40, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"File: {file_name}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Sauvegarde image
        output_name = f"Analyse 1 - avec export csv - {file_name}"
        cv2.imwrite(output_name, frame)

print("\nTraitement terminé avec succès.")
print(f"Résultats CSV sauvegardés dans : {CSV_PATH}")

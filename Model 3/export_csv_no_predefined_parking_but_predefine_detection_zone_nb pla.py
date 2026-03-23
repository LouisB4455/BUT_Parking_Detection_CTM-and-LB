import cv2
import pickle
import numpy as np
import os
import glob
import csv
from ultralytics import YOLO

# --- CONFIGURATION ---
FOLDER_PATH = "photo/"
ZONE_PATH = "detection_zone_2.pkl"
MODEL_PATH = "yolov8m.pt"
CSV_PATH = "results_zone.csv"

# Chargement modèle YOLO
model = YOLO(MODEL_PATH)

# Chargement zone + capacité
with open(ZONE_PATH, "rb") as f:
    data = pickle.load(f)
    detection_zone = data["zone"]
    capacity = data["capacity"]

zone_poly = np.array(detection_zone, np.int32)

# Lister toutes les images
image_extensions = ['*.jpg', '*.jpeg', '*.png']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))

if not image_files:
    print(f"Aucune image trouvée dans {FOLDER_PATH}")
    exit()

print(f"Début de l'analyse : {len(image_files)} images à traiter.")

# --- CSV INIT ---
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "cars_in_zone", "free_places", "capacity"])

    # --- BOUCLE DE TRAITEMENT ---
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Erreur de lecture : {img_path}")
            continue

        file_name = os.path.basename(img_path)

        # Détection YOLO
        results = model(frame, conf=0.35, verbose=False)

        cars_in_zone = 0

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 2:  # voiture
                    x1, y1, x2, y2 = box.astype(int)

                    # centre voiture
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # test zone
                    if cv2.pointPolygonTest(zone_poly, (cx, cy), False) >= 0:
                        cars_in_zone += 1
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

        # calcul
        free_places = max(0, capacity - cars_in_zone)

        # écriture CSV
        writer.writerow([file_name, cars_in_zone, free_places, capacity])

        # dessin zone
        cv2.polylines(frame, [zone_poly], True, (255, 0, 0), 3)

        # cartouche
        cv2.rectangle(frame, (20, 20), (600, 190), (0, 0, 0), -1)

        cv2.putText(frame, f"Cars in zone: {cars_in_zone}", (40, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Free: {free_places} / {capacity}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"File: {file_name}", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # sauvegarde image
        output_name = f"Analyse 3 - {file_name}"
        cv2.imwrite(output_name, frame)

        print(f"Enregistré : {output_name}")

print("\nTraitement terminé avec succès.")
print(f"CSV sauvegardé : {CSV_PATH}")

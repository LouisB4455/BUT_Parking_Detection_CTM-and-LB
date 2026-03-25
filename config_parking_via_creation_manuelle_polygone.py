import cv2
import pickle
import os
import numpy as np

image_path = "Model 1/image_de_depart_pour_analyse/2026-03-03_1134.jpg"
pickle_path = "parking_slots.pkl"

# chargement des places existantes
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        pos_list = pickle.load(f)
else:
    pos_list = []

current_points = []

def mouse_click(event, x, y, flags, params):

    global current_points, pos_list

    # ajouter un point
    if event == cv2.EVENT_LBUTTONDOWN:

        current_points.append((x, y))

        # si 4 points -> créer la place
        if len(current_points) == 4:

            pos_list.append(current_points.copy())
            current_points = []

            with open(pickle_path, "wb") as f:
                pickle.dump(pos_list, f)

    # clic droit -> supprimer une place
    if event == cv2.EVENT_RBUTTONDOWN:

        for i, poly in enumerate(pos_list):

            pts = np.array(poly, np.int32)

            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                pos_list.pop(i)
                break

        with open(pickle_path, "wb") as f:
            pickle.dump(pos_list, f)


cv2.namedWindow("Preparation Parking", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Preparation Parking", mouse_click)

while True:

    img = cv2.imread(image_path)

    if img is None:
        print("Image introuvable")
        break

    # dessiner les places existantes
    for poly in pos_list:

        pts = np.array(poly, np.int32)

        cv2.polylines(img, [pts], True, (255,0,255), 2)

    # dessiner les points en cours
    for p in current_points:

        cv2.circle(img, p, 5, (0,255,255), -1)

    if len(current_points) > 1:
        cv2.polylines(img, [np.array(current_points)], False, (0,255,255), 2)

    cv2.imshow("Preparation Parking", img)

    key = cv2.waitKey(1)

    # quitter et sauvegarder
    if key == ord("s"):
        with open(pickle_path, "wb") as f:
            pickle.dump(pos_list, f)
        break

    # reset points en cours
    if key == ord("r"):
        current_points = []

cv2.destroyAllWindows()

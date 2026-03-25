import argparse
import os
import pickle

import cv2
import numpy as np


def load_existing(path: str):
    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "zones" in data:
        return data["zones"]

    raise ValueError("Format non supporte pour zones interdites")


def save_zones(path: str, zones):
    with open(path, "wb") as f:
        pickle.dump({"zones": zones}, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration manuelle des zones interdites")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="zones_interdites.pkl")
    return parser.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Image introuvable: {args.image}")
        return

    zones = load_existing(args.output)
    current = []

    def on_mouse(event, x, y, flags, params):
        nonlocal current, zones

        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, poly in enumerate(zones):
                pts = np.array(poly, np.int32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    zones.pop(i)
                    break

    cv2.namedWindow("Config Zones Interdites", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Config Zones Interdites", on_mouse)

    while True:
        view = img.copy()

        for poly in zones:
            pts = np.array(poly, np.int32)
            cv2.polylines(view, [pts], True, (0, 0, 255), 2)

        for p in current:
            cv2.circle(view, p, 4, (0, 255, 255), -1)

        if len(current) > 1:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 255, 255), 2)

        cv2.rectangle(view, (10, 10), (920, 110), (0, 0, 0), -1)
        cv2.putText(view, "Left click: ajouter point | N: valider zone | R: reset zone", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, "Right click: supprimer zone | S: sauvegarder et quitter", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Config Zones Interdites", view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            if len(current) >= 3:
                zones.append(current.copy())
                current = []

        elif key == ord("r"):
            current = []

        elif key == ord("s"):
            save_zones(args.output, zones)
            print(f"Sauvegarde: {args.output} ({len(zones)} zones)")
            break

        elif key == ord("q"):
            print("Quitte sans sauvegarde")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

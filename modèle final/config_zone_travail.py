import argparse
import os
import pickle
import re

import cv2
import numpy as np


def normalize_zone(zone):
    if not isinstance(zone, list):
        return []
    out = []
    for pt in zone:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append((int(pt[0]), int(pt[1])))
    return out if len(out) >= 3 else []


def load_existing_profiles(path: str):
    if not os.path.exists(path):
        return {}, None

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "zone" in data:
        ref = data.get("reference_image")
        return {
            "default": {
                "name": "default",
                "zone": normalize_zone(data.get("zone", [])),
                "reference_image": os.path.abspath(ref) if isinstance(ref, str) else None,
            }
        }, "default"

    if isinstance(data, list):
        return {
            "default": {
                "name": "default",
                "zone": normalize_zone(data),
                "reference_image": None,
            }
        }, None

    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list):
        profiles = {}
        for i, p in enumerate(data["profiles"]):
            if not isinstance(p, dict):
                continue
            name = p.get("name") if isinstance(p.get("name"), str) else f"profile_{i+1}"
            ref = p.get("reference_image")
            profiles[name] = {
                "name": name,
                "zone": normalize_zone(p.get("zone", [])),
                "reference_image": os.path.abspath(ref) if isinstance(ref, str) else None,
            }
        active = data.get("active_profile") if isinstance(data.get("active_profile"), str) else None
        return profiles, active

    raise ValueError("Format non supporte pour zone de travail")


def save_profiles(path: str, profiles: dict, active_profile: str):
    if len(profiles) == 1:
        one = next(iter(profiles.values()))
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "zone": one.get("zone", []),
                    "reference_image": one.get("reference_image"),
                },
                f,
            )
        return

    payload = {
        "active_profile": active_profile,
        "profiles": [],
    }
    for name in sorted(profiles.keys()):
        p = profiles[name]
        payload["profiles"].append(
            {
                "name": p.get("name", name),
                "zone": p.get("zone", []),
                "reference_image": p.get("reference_image"),
            }
        )

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def infer_profile_name(image_path: str):
    basename = os.path.basename(image_path)
    m = re.search(r"(19|20)\d{2}", basename)
    if m:
        return m.group(0)
    m = re.search(r"(19|20)\d{2}", image_path)
    if m:
        return m.group(0)
    return "default"


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration manuelle de la zone de travail")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="parking_zone.pkl")
    parser.add_argument("--profile-name", default=None, help="Nom de profil (ex: 2024, 2026)")
    return parser.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Image introuvable: {args.image}")
        return

    h, w = img.shape[:2]
    profiles, _ = load_existing_profiles(args.output)
    profile_name = args.profile_name.strip() if isinstance(args.profile_name, str) and args.profile_name.strip() else infer_profile_name(args.image)

    if profile_name not in profiles:
        profiles[profile_name] = {
            "name": profile_name,
            "zone": [],
            "reference_image": os.path.abspath(args.image),
        }

    if not profiles[profile_name].get("reference_image"):
        profiles[profile_name]["reference_image"] = os.path.abspath(args.image)

    current = profiles[profile_name].get("zone", []).copy()
    print(f"Profil actif: {profile_name} | zone existante: {'oui' if len(current) >= 3 else 'non'}")

    def on_mouse(event, x, y, flags, params):
        nonlocal current

        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current:
                current.pop()

    cv2.namedWindow("Config Zone Travail", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Config Zone Travail", on_mouse)

    while True:
        view = img.copy()

        if len(current) >= 3:
            pts = np.array(current, np.int32)
            cv2.polylines(view, [pts], True, (255, 0, 0), 2)

        for p in current:
            cv2.circle(view, p, 4, (0, 255, 255), -1)

        if len(current) > 1:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 255, 255), 2)

        cv2.rectangle(view, (10, 10), (960, 125), (0, 0, 0), -1)
        cv2.putText(view, f"Image originale: {w}x{h}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(view, f"Profil work zone: {profile_name}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(view, "Left click: ajouter point | Right click: retirer dernier point", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(view, "R: reset | S: sauvegarder | E: sauver vide | Q: quitter", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Config Zone Travail", view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("r"), ord("R")):
            current = []

        elif key in (ord("s"), ord("S")):
            zone = normalize_zone(current)
            if len(zone) < 3:
                print("Zone invalide. Ajoute au moins 3 points, ou appuie sur E pour sauver vide.")
                continue
            profiles[profile_name]["zone"] = zone
            profiles[profile_name]["reference_image"] = os.path.abspath(args.image)
            save_profiles(args.output, profiles, profile_name)
            print(f"Sauvegarde: {args.output} | profil={profile_name} (points={len(zone)})")
            break

        elif key in (ord("e"), ord("E")):
            profiles[profile_name]["zone"] = []
            profiles[profile_name]["reference_image"] = os.path.abspath(args.image)
            save_profiles(args.output, profiles, profile_name)
            print(f"Sauvegarde vide forcee: {args.output} | profil={profile_name}")
            break

        elif key in (ord("q"), ord("Q")):
            print("Quitte sans sauvegarde")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

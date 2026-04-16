"""
Analyse le dossier DATA pour détecter quand la caméra a changé de position/angle.
Compare les images d'une date à l'autre pour identifier les changements significatifs.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

DATA_FOLDER = r"C:\Users\cdeth\OneDrive - ISEP\Documents\Projet ISEP VUT\BUT_Parking_Detection_CTM-and-LB\DATA"

# Récupère toutes les dates
dates = sorted([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))])

print(f"\n{'='*80}")
print(f"ANALYSE CAMÉRA - {len(dates)} dates trouvées")
print(f"{'='*80}\n")

# Prendre la première image de chaque date
images_by_date = {}
for date in dates:
    date_path = os.path.join(DATA_FOLDER, date)
    images = sorted(glob.glob(os.path.join(date_path, "*.jpg")))
    if images:
        images_by_date[date] = images[0]
        print(f"{date}: {os.path.basename(images[0])}")

print(f"\n{'='*80}")
print("DÉTECTION DES CHANGEMENTS DE CAMÉRA")
print(f"{'='*80}\n")

# Comparer des images consécutives avec ORB
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

prev_date = None
prev_img = None
significant_changes = []

for date in dates:
    if date not in images_by_date:
        continue
    
    img_path = images_by_date[date]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"⚠️  {date}: Impossible de lire l'image")
        continue
    
    # Redimensionner pour accélérer (optionnel)
    img_small = cv2.resize(img, (800, 450))
    
    if prev_img is not None:
        # Détecter les keypoints
        kp1, des1 = orb.detectAndCompute(prev_img, None)
        kp2, des2 = orb.detectAndCompute(img_small, None)
        
        if des1 is not None and des2 is not None:
            # Matcher les features
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Appliquer Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            match_ratio = len(good_matches) / max(len(matches), 1)
            
            print(f"{prev_date} → {date}")
            print(f"  Matches: {len(good_matches)}/{len(matches)} ({match_ratio*100:.1f}%)")
            
            # Si peu de bonnes correspondances = changement de perspective
            if match_ratio < 0.3 or len(good_matches) < 20:
                print(f"  ⚠️  CHANGEMENT DÉTECTÉ !")
                significant_changes.append((prev_date, date))
            else:
                print(f"  ✓ Caméra stable")
            print()
    
    prev_date = date
    prev_img = img_small

print(f"\n{'='*80}")
print("RÉSUMÉ")
print(f"{'='*80}\n")

if significant_changes:
    print(f"🔴 {len(significant_changes)} changement(s) de caméra détecté(s):\n")
    for prev, curr in significant_changes:
        print(f"  • Entre {prev} et {curr}")
else:
    print("✅ Aucun changement significatif détecté")
    print("   → La caméra est restée au même endroit/angle pour tout le dataset")

print(f"\n{'='*80}")

import cv2
import pickle

# Check reference line image
with open("modèle final/ligne_reference.pkl", "rb") as f:
    ref_data = pickle.load(f)
ref_img_path = ref_data['reference_image']
print(f"Reference image path: {ref_img_path}")

ref_img = cv2.imread(ref_img_path)
if ref_img is not None:
    h, w = ref_img.shape[:2]
    print(f"Reference image size: {w}x{h}")
else:
    print("Could not load reference image")

# Check forbidden zones reference image
print("\nChecking forbidden zones...")
with open("modèle final/zones_interdites.pkl", "rb") as f:
    zones_data = pickle.load(f)
zones_ref_img = zones_data.get('reference_image')
if zones_ref_img:
    print(f"Zones reference image: {zones_ref_img}")
    zone_img = cv2.imread(zones_ref_img)
    if zone_img is not None:
        h, w = zone_img.shape[:2]
        print(f"Zones reference image size: {w}x{h}")

# Check a test image from 2024-04-05
print("\nTest image from 2024-04-05:")
test_img = cv2.imread("DATA/2024-04-05/2024-04-05 17_04_26.jpg")
if test_img is not None:
    h, w = test_img.shape[:2]
    print(f"Test image size: {w}x{h}")

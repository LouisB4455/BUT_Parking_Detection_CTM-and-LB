import pickle
import os

os.chdir(r"c:\Users\cdeth\OneDrive - ISEP\Documents\Projet ISEP VUT\BUT_Parking_Detection_CTM-and-LB\modèle final")

# Check reference line
print("=" * 60)
print("REFERENCE LINE CHECK")
print("=" * 60)
if os.path.exists('ligne_reference.pkl'):
    with open('ligne_reference.pkl', 'rb') as f:
        ref_data = pickle.load(f)
    print('Reference line data:')
    print(f'  Type: {type(ref_data)}')
    if isinstance(ref_data, dict):
        print(f'  Keys: {ref_data.keys()}')
        for key, val in ref_data.items():
            if key == 'line' and isinstance(val, (list, tuple)):
                print(f'    {key}: {len(val)} points')
                for i, pt in enumerate(val):
                    print(f'      Point {i}: {pt}')
            else:
                print(f'    {key}: {val}')
    else:
        print(f'  Content (first 50 chars): {str(ref_data)[:50]}')
else:
    print('Reference line file NOT FOUND')

print()
print("=" * 60)
print("FORBIDDEN ZONES CHECK")
print("=" * 60)

# Check forbidden zones
if os.path.exists('zones_interdites.pkl'):
    with open('zones_interdites.pkl', 'rb') as f:
        zones_data = pickle.load(f)
    print('Forbidden zones data:')
    print(f'  Type: {type(zones_data)}')
    if isinstance(zones_data, list):
        print(f'  Number of zones: {len(zones_data)}')
        if len(zones_data) > 0:
            zone = zones_data[0]
            print(f'  Zone 0: {len(zone)} points')
            print(f'    First 3 points: {zone[:3]}')
    elif isinstance(zones_data, dict):
        print(f'  Keys: {zones_data.keys()}')
        if 'zones' in zones_data:
            zones_list = zones_data['zones']
            print(f'  Number of zones: {len(zones_list)}')
            if len(zones_list) > 0:
                print(f'  Zone 0: {len(zones_list[0])} points')
else:
    print('Zones file NOT FOUND')

print()
print("=" * 60)
print("SAMPLE IMAGE CHECK")
print("=" * 60)
DATA_folder = r"../DATA/2024-04-05"
if os.path.exists(DATA_folder):
    import glob
    images = sorted(glob.glob(os.path.join(DATA_folder, "*.jpg")) + glob.glob(os.path.join(DATA_folder, "*.png")))
    print(f"Found {len(images)} images in {DATA_folder}")
    if images:
        print(f"First image: {os.path.basename(images[0])}")
else:
    print(f"DATA folder NOT FOUND: {DATA_folder}")

# Correction: Coordinate System Alignment

## Problem Identified
The reference line and forbidden zones were drawn on a **2026-02-24 image** (small canvas ~1000×670px) but the test dataset **2024-04-05** has large images (3200×1800px). This caused massive coordinate misalignment.

## Solution Applied
**Date**: Current session
**Actions**:
1. Redrew reference line on correct 2024-04-05 image ✅
   - File: `ligne_reference.pkl`
   - Reference line: `((74.0, 539.0), (1155.0, 563.0))` in 3200×1800 coordinate frame
   - Reference image: `DATA/2024-04-05/2024-04-05 17_04_26.jpg`

2. Redrew forbidden zones on same 2024-04-05 image ✅
   - File: `zones_interdites.pkl`
   - 1 forbidden zone polygon in matching coordinate frame

3. Updated `interface_selection_data.py` ✅
   - Added `--line-reference ligne_reference.pkl` 
   - Added `--forbidden-zones zones_interdites.pkl`
   - Now correctly passes file paths to `analyse_modele_final.py`

## Verification
- Pipeline tested on 2024-04-05 dataset (6 frames)
- Output samples show correct zone alignment with detected vehicles
- dx/dy = 0.0 (no spurious camera movement detected)
- Count results coherent: 4-9 cars total, 1-3 in forbidden zone

## Key Lesson
**Reference and test data must use the same coordinate system.** When redrawing calibration data, ensure the reference image resolution matches the test dataset resolution.

## Files Modified
- `modèle final/interface_selection_data.py` (args added)
- `modèle final/ligne_reference.pkl` (redrawn)
- `modèle final/zones_interdites.pkl` (redrawn)

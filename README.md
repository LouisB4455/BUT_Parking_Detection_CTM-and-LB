# BUT_Parking_Detection_CTM-and-LB

Projet de detection et comptage de voitures sur parking, organise autour de deux usages actifs:
- la pipeline d'analyse et de monitoring locale,
- la pipeline d'entrainement batch YOLO.

Les anciens blocs legacy ont ete retires du workspace actif. La documentation a jour se trouve ici et dans `modèle final/README.md`.

## Arborescence active

### `DATA/`
Dossier source des images et des labels.
- `DATA/` contient les captures par date pour l'analyse.
- `DATA/DATA_1` et `DATA/LABELS_1` servent de base au dataset YOLO.

### `modèle final/`
Dossier principal du projet actif.
- code de pipeline,
- code d'entrainement,
- code de calibration / alignement,
- sorties et rapports generes localement.

### `tools/`
Dossier des scripts optionnels (maintenance, tuning, validation avancee).

## Scripts Python de `modèle final/`

### Pipeline d'analyse

- `analyse_modele_final.py`: lit des sous-dossiers `DATA`, lance YOLO, compte les voitures et ecrit les CSV / images annotées.
- `interface_selection_data.py`: interface graphique pour choisir les sous-dossiers `DATA` et lancer la pipeline d'analyse.
- `mettre_a_jour_monitoring_html.py`: reconstruit ou rafraichit le dashboard HTML local a partir des CSV.

### Definition des zones

- `config_zones_interdites.py`: outil interactif pour dessiner et sauvegarder les zones interdites.
- `config_zone_travail.py`: outil interactif pour dessiner et sauvegarder la zone de travail / parking.

### Preparation des donnees YOLO

- `prepare_yolo_dataset.py`: cree un dataset YOLO `train/val/test` et genere `dataset.yaml`.
- `offline_augment_dataset.py`: produit des variantes augmentees hors ligne a partir des images/labels source.

### Entrainement YOLO

- `train_batch_yolo.py`: entraine un modele YOLO sur le dataset prepare et ecrit un rapport JSON + HTML.

### Outils optionnels (`tools/`)

- `tools/run_training_grid_3runs.py`: wrapper racine pour lancer la grille 3 runs depuis `tools/training/`.
- `tools/training/run_training_grid_3runs.py`: compare 3 configurations d'entrainement et classe les resultats.
- `tools/training/validate_batch_model.py`: applique un modele entraine a des images inconnues et exporte un CSV de synthese.
- `tools/training/verify_data.py`: verification rapide de structure DATA injectee dans le monitoring HTML.
- `tools/alignment/alignment_ml_utils.py`: fonctions communes du module d'alignement camera.
- `tools/alignment/generate_alignment_synth_dataset.py`: generation de dataset synthetique pour l'alignement.
- `tools/alignment/train_alignment_offset_model.py`: entrainement du modele lineaire d'alignement.
- `tools/alignment/parking_grid_homography.py`: detection de lignes / homographie de grille.
- `tools/alignment/calibrate_grid_corners.py`: outil interactif de calibration de coins.
- `tools/alignment/test_corner_detection.py`: test visuel de detection des coins.

### Sorties et launchers

- `run_batch_training.bat`: lance le flux complet preparation + entrainement batch.
- `run_pipeline_complet.bat`: lance la pipeline d'analyse depuis Windows.
- `README.md`: documentation operative du dossier `modèle final/`.

## Dossiers generes / artefacts locaux

Ces dossiers contiennent des sorties de train ou de pipeline, pas du code applicatif.
- `modèle final/batch_dataset/`: dataset YOLO prepare.
- `modèle final/resultats_modele_final/`: images annotées de la pipeline.
- `modèle final/training_runs/`: sorties Ultralytics des runs d'entrainement.
- `modèle final/runs/`: sorties Ultralytics supplementaires / historiques.
- `modèle final/yolo_correction_dataset/`: ancien dataset de correction genere localement.

## Lancement rapide

- Pipeline analyse + monitoring: `modèle final/run_pipeline_complet.bat`
- Entrainement batch: `modèle final/run_batch_training.bat`

## Notes

- Les rapports grid generes localement ne sont plus conserves dans le depot.
- La documentation de fonctionnement detaillee reste dans `modèle final/README.md`.

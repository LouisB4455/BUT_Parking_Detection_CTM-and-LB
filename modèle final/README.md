# Modele Final (v2)

Pipeline simplifie:
- detection de voitures
- comptage total
- comptage voitures en zone interdite
- comptage voitures legales = total - interdites
- affichage des zones statiques: rouge (interdite) + bleu (zone parking)

## Fichiers principaux

- analyse_modele_final.py: pipeline principal
- config_zones_interdites.py: edition des zones interdites
- parking_zone.pkl: zone de travail/parking (zone bleue)
- mettre_a_jour_monitoring_html.py: injecte les donnees dans le HTML
- interface_selection_data.py: interface locale de selection des sous-dossiers DATA
- monitoring_final_simple.html: dashboard local (sans serveur)
- prepare_yolo_dataset.py: preparation automatique du dataset YOLO (train/val/test + dataset.yaml)
- offline_augment_dataset.py: augmentation hors-ligne (10 variations par image)
- train_batch_yolo.py: entrainement transfer learning sur dataset prepare
- ../tools/training/validate_batch_model.py: validation batch sur images inconnues et export des resultats

## Prerequis

Bibliotheques Python:
- opencv-python
- numpy
- ultralytics

## 1) Definir la zone interdite

Exemple:

python config_zones_interdites.py --image "../DATA/2026-03-03/2026-03-03_1134.jpg" --output zones_interdites.pkl

Touches:
- clic gauche: ajouter un point
- N: valider la zone en cours
- S: sauvegarder et quitter
- R: reset zone en cours
- clic droit dans une zone: supprimer la zone
- Q: quitter sans sauvegarder

## 2) Lancer l'analyse

Tous les dossiers DATA:

python analyse_modele_final.py --input-folder "../DATA"

Sous-dossiers selectionnes:

python analyse_modele_final.py --input-folder "../DATA" --include-subfolders "2026-03-03" "2026-03-04"

## 3) Interface graphique (option 1)

Lancer:

run_pipeline_complet.bat

L'interface permet de:
- choisir les sous-dossiers DATA
- lancer l'analyse
- regenerer le monitoring HTML

## 4) Batch Training (nouvelle methode)

Pipeline d'entrainement recommande:

1. Augmentation hors-ligne (optionnel):

python offline_augment_dataset.py --images-dir "../DATA/DATA_1" --labels-dir "../DATA/LABELS_1" --output-images-dir "../DATA/DATA_1_AUG" --output-labels-dir "../DATA/LABELS_1_AUG" --variations 10

2. Preparation train/val/test au format YOLO:

python prepare_yolo_dataset.py --images-dir "../DATA/DATA_1" --labels-dir "../DATA/LABELS_1" --output-dir "batch_dataset" --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --class-names "car"

3. Entrainement transfer learning:

python train_batch_yolo.py --data "batch_dataset/dataset.yaml" --weights "yolov8n.pt" --epochs 100 --imgsz 640 --batch 16

Le meilleur modele est ensuite publie automatiquement vers `parking_detector_corrections.pt`.
Si ce fichier existait deja, il est remplace par la nouvelle version.

Rapport de fin d'entrainement genere automatiquement:
- training_batch_last_report.json
- training_batch_last_report.html
- le rapport indique aussi si l'ancien modele a ete remplace

Indices suivis dans le rapport:
- precision train
- precision val
- recall train
- recall val
- mAP50 train
- mAP50 val
- mAP50-95 train
- mAP50-95 val
- train/val box_loss, cls_loss, dfl_loss

4. Validation batch sur images inconnues:

python ../tools/training/validate_batch_model.py --model "training_runs/batch_yolo/weights/best.pt" --images-dir "../DATA_UNKNOWN" --output-dir "validation_outputs"

Lancement rapide entrainement:

run_batch_training.bat

## Monitoring

- Ouvrir monitoring_final_simple.html
- Rafraichir les donnees:

python mettre_a_jour_monitoring_html.py

## Sorties


Colonnes principales utilisees pour l'export JSON / CSV:
- image_name
- timestamp
- Nb voitures
- Nb voitures légales
- places_libres
- places_occupees
- places_hors
- temps_de_traitement
- username

Colonnes CSV principales:
- cars_in_forbidden

## Notes

- Le nettoyage des images de sortie est cible: un rerun d'un sous-dossier supprime
  uniquement les anciennes sorties de ce sous-dossier.
- La zone interdite est tracee en rouge sur les images.
- La zone de travail (parking) est tracee en bleu sur les images.
- Les artefacts grid3 generes localement ne sont plus conserves dans le depot.

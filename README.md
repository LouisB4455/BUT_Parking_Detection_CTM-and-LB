# BUT_Parking_Detection_CTM-and-LB

Projet de detection et comptage de voitures sur parking, avec:
- pipeline d'analyse de sous-dossiers DATA,
- monitoring HTML local,
- pipeline d'entrainement batch YOLO.

## Point d'entree principal

Le projet actif est dans `modèle final/`.

Documentation principale:
- `modèle final/README.md`

## Lancement rapide

- Pipeline analyse + monitoring:
	- `modèle final/run_pipeline_complet.bat`
- Entrainement batch:
	- `modèle final/run_batch_training.bat`

## Donnees

- Donnees d'analyse: `DATA/` (sous-dossiers dates)
- Donnees d'entrainement YOLO source: `DATA/DATA_1` et `DATA/LABELS_1`

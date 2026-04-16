# Guide interactif - correction_gui.py

## Important

La GUI actuelle est basee sur Tkinter, pas sur OpenCV plein ecran.
Les raccourcis clavier `L/I/S/N/Q` de l'ancienne version ne s'appliquent plus.

## Principe

Pour chaque image de `manual_review_queue.txt`:
- choisir un mode de correction a droite
- cliquer sur l'image (clic gauche)
- verifier les valeurs corrigees
- passer a l'image suivante ou terminer

## Modes disponibles

- `Legale (vert)`: ajoute une voiture legale
- `Illegale (rouge)`: ajoute une voiture illegale
- `Non detectee (bleu)`: ajoute une voiture manquee (comptee comme legale)
- `Faux positif (gris)`: retire une detection (retire d'abord une illegale si possible)

## Workflow recommande

1. Lancer la file priorisee:

```bash
run_correction_manuelle_priorisee.bat
```

2. Dans la GUI:
- verifier l'image
- selectionner le mode
- cliquer sur les zones a corriger
- utiliser `Supprimer 1 voiture` pour annuler le dernier ajout si besoin
- cliquer `Suivante` pour continuer

3. A la fin:
- cliquer `TERMINER & QUITTER` pour ecrire toutes les corrections

## Fichiers mis a jour

- `resultats_modele_final.csv`: compteurs corriges
- `manual_review_done.csv`: resume des images corrigees
- `manual_review_annotations.json`: points de correction pour reentrainement

## Conseils pratiques

- corriger en priorite les faux positifs et les voitures manquees
- utiliser des boites proposees coherentes (sliders largeur/hauteur)
- finaliser la session avec `TERMINER & QUITTER` pour garantir la persistence

# Outil de correction manuelle GUI

## Version actuelle

Ce document décrit la GUI actuelle (`correction_gui.py`):
- interface Tkinter
- correction par clic sur l'image
- pas de champs texte "Valider/Passer" (ancienne doc)
- sauvegarde des corrections dans CSV et JSON

## Workflow

1. `analyse_modele_final.py` produit `resultats_modele_final.csv`
2. `preparer_lot_correction.py` construit `manual_review_queue.txt`
3. `correction_gui.py` applique les corrections manuelles
4. `mettre_a_jour_monitoring_html.py` met a jour `monitoring_final_simple.html`

## Interface

### Panneau gauche
- image a corriger (image annotee si disponible, sinon image brute)

### Panneau droit
- valeurs originales: total / illegales / legales
- mode d'action:
  - `legal`
  - `illegal`
  - `missed` (voiture non detectee)
  - `false_positive` (faux positif a retirer)
- sliders de taille de boite proposee
- actions: supprimer 1 voiture, reinitialiser
- valeurs corrigees recalculées en temps reel
- navigation: precedente / suivante / terminer

## Regles de correction

- `legal`: +1 voiture legale
- `illegal`: +1 voiture illegale
- `missed`: +1 voiture legale
- `false_positive`: -1 voiture (d'abord illegale, sinon legale)

Le total corrige est toujours `legales + illegales`.

## Fichiers utilises

### Entree
- `manual_review_queue.txt`
- `resultats_modele_final.csv`

### Sortie
- `resultats_modele_final.csv` (compteurs corriges)
- `manual_review_done.csv` (resume par image)
- `manual_review_annotations.json` (points de correction, utile pour le training)

## Lancement

Depuis le dossier `modele final`:

```bash
python correction_gui.py
```

Ou via l'orchestrateur:

```bash
python interface_selection_data.py
```

## Notes

- la sauvegarde finale est faite avec `TERMINER & QUITTER`
- `Suivante` conserve l'etat en memoire
- `Quitter` ferme sans finaliser le lot complet

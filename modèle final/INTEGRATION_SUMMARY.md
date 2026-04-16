# Intégration de l'Outil de Correction GUI

## Résumé des changements

### ✅ Fichiers créés

1. **`correction_gui.py`** (398 lignes)
   - Interface graphique complète avec Tkinter
   - Affichage des images avec détails de détection
   - Zones de correction pour total_cars, cars_in_forbidden, cars_legal
   - Sauvegarde automatique des corrections dans le CSV
   - Gestion de la navigation (précédent/suivant)
   - Système de notes optionnelles
   - Barre de progression

2. **`run_pipeline_complet.bat`**
   - Script de lancement du pipeline complet
   - Facilite l'exécution depuis Windows

3. **`CORRECTION_GUI_README.md`**
   - Documentation complète de l'outil
   - Architecture et intégration
   - Guide d'utilisation
   - Spécifications techniques

### ✅ Fichiers modifiés

1. **`lancer_correction_manuelle.py`**
   - Remplacé le simple prompt par le lancement du GUI
   - Appelle `correction_gui.py` via subprocess

2. **`interface_selection_data.py`**
   - Ajout de l'étape "Preparation du lot de correction"
   - Appelle `preparer_lot_correction.py` après l'analyse
   - Puis lance la correction GUI via `run_correction_manuelle.bat`

## Workflow intégré

```
┌─────────────────────────────────
│ interface_selection_data.py
│ (Sélection des dossiers)
└──────────────┬──────────────────
               ↓
┌─────────────────────────────────
│ analyse_modele_final.py  
│ (YOLOv8 - Détection voitures)
└──────────────┬──────────────────
               ↓
┌─────────────────────────────────
│ preparer_lot_correction.py
│ (Prioriser images à corriger)
│ Génère: manual_review_queue.txt
└──────────────┬──────────────────
               ↓
┌─────────────────────────────────
│ correction_gui.py
│ (GUI - Correction manuelle)
│ Met à jour: resultats_modele_final.csv
└──────────────┬──────────────────
               ↓
┌─────────────────────────────────
│ mettre_a_jour_monitoring_html.py
│ (Actualiser monitoring.html)
└─────────────────────────────────
```

## Utilisation

### Par l'interface principale
1. Lancer `interface_selection_data.py`
2. Sélectionner les sous-dossiers à analyser
3. Cliquer "Lancer Modele Sur Selection"
4. Le workflow complet s'exécute automatiquement

### Par le batch
```bash
run_pipeline_complet.bat
```

### Directement la correction
```bash
python correction_gui.py
```
(Nécessite que `manual_review_queue.txt` existe)

## Interface GUI - Guide rapide

**Navigation:**
- Flèches temporelles: < Précédent | Suivant >
- Boutons d'action: Passer | Quitter | Valider

**Correction:**
1. Examiner l'image affichée
2. Ajuster les counts si nécessaire (Total, Interdites, Légales)
3. Cliquer "Valider" pour sauvegarder et continuer
4. Cliquer "Passer" pour ignorer

**Barre de progression:**
- Position: "Image X/Y"
- Statut: "N images corrigées / Total"

## Données côté données

### Fichiers en entrée
- `manual_review_queue.txt` – Liste prioritaire d'images
- `resultats_modele_final.csv` – Résultats de l'analyse

### Fichiers en sortie
- `resultats_modele_final.csv` – MISE À JOUR avec corrections
- `manual_review_done.csv` – Log des images corrigées

## Dépendances

- Python 3.11+
- Pillow (>= 10.0) – Pour affichage images
- Tkinter (inclus in Python)

Toutes les dépendances sont déjà installées dans le `.venv` du projet.

## Notes techniques

- Les corrections sont **sauvegardées immédiatement** après validation
- Les chemins images sont normalisés (séparateurs `/`)
- Gestion des rôles:
  - `preparer_lot_correction.py` – Génère la priorité
  - `correction_gui.py` – Interface utilisateur
  - Reste du système – Sauvegarde et intégration

## Prochaines étapes

- Si nécessaire: ajouter un mode "expert" avec annotation de zones
- Intégration monitoring en temps réel
- Export statistiques de correction


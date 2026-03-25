# Modele Final (v1)

Premiere version qui combine:
- recalage automatique si la camera bouge legerement
- detection place libre/occupee par polygones de places
- detection de stationnement illegal hors places officielles

## Fichiers

- analyse_modele_final.py: pipeline principal
- config_zones_interdites.py: outil manuel pour definir les zones interdites

## Prerequis

Bibliotheques Python:
- opencv-python
- numpy
- ultralytics

## Etape 1 - Configurer les zones interdites (optionnel mais recommande)

Exemple:

python config_zones_interdites.py --image "../Model 1/image_de_depart_pour_analyse/2026-03-03_1034.jpg" --output zones_interdites.pkl

Commandes dans la fenetre:
- clic gauche: ajouter un point
- touche N: valider une zone (>= 3 points)
- clic droit dans une zone: supprimer la zone
- touche R: annuler zone en cours
- touche S: sauvegarder

## Etape 2 - Lancer l'analyse

Exemple:

python analyse_modele_final.py \
  --input-folder "../Model 1/image_de_depart_pour_analyse" \
  --parking-slots "../Model 1/parking_slots.pkl" \
  --parking-zone "../Model 3/detection_zone_2.pkl" \
  --forbidden-zones "zones_interdites.pkl" \
  --output-folder "resultats_modele_final" \
  --csv-path "resultats_modele_final.csv"

## Sorties

- resultats_modele_final.csv:
  - free_places
  - occupied_places
  - cars_detected
  - illegal_parked
  - indicateurs de recalage (matches/inliers)
- resultats_modele_final/*.jpg: images annotees

## Monitoring (interface statistiques)

Version HTML autonome (sans serveur):

- Ouvrir directement le fichier monitoring_officiel.html dans un navigateur.
- Cette page contient:
  - un onglet monitoring principal
  - un onglet monitoring erreurs
  - les KPI temps reel (base image la plus recente)
  - legendes visibles sur les graphes
- Le bouton "Rafraichir la page" lance le script run_modele_et_refresh.bat
  - nettoyage check manuel
  - lancement modele final
  - regeneration du HTML
  - re-ouverture de la page

Fichier:

monitoring_officiel.html

Important (mode site web HTML):

- En mode navigateur, les scripts locaux (.bat/.py) ne peuvent pas etre executes directement (securite navigateur).
- Les boutons affichent donc le script a lancer manuellement.
- Interface 100% HTML conservee (sans serveur, sans HTA obligatoire).

Lancer le serveur local de monitoring officiel (sans input):

python monitoring_server.py

Puis ouvrir:

http://127.0.0.1:8050

Fonctionnalites:
- KPI temps reel (image la plus recente):
  - pourcentage de places occupees
  - nombre de places occupees
  - nombre de places libres
  - nombre de voitures en place illegale
- courbe d'occupation
- evolution des places occupees/libres
- stats du check manuel (err1..err10) + ecarts avec la sortie modele

Boutons integres (page principale):
- Page Monitoring Erreurs
- Lancer Correction Manuelle
- Lancer Modele + Rafraichir
- Rafraichir Maintenant

Page erreurs:
- URL: http://127.0.0.1:8050/errors
- totaux des erreurs manuelles
- detail par image
- bouton pour lancer la correction manuelle

Notes monitoring:
- aucun champ de saisie
- lecture automatique de resultats_modele_final.csv et check_manuel_results.csv

## Notes

- Le recalage utilise ORB + RANSAC.
- Si le recalage echoue sur une image, le script retombe sur les polygones de reference.
- Le critere illegal est base sur:
  - overlap avec zone interdite
  - overlap trop faible avec toute place officielle dans la zone parking

# 🚗 BUT Parking Detection - Guide d'Installation

## Prérequis
- **Python 3.9+** (testé avec Python 3.11)
- **Git**
- **VS Code** (optionnel mais recommandé)

## Installation en 5 minutes

### 1. Cloner le repo
```powershell
git clone https://github.com/TON-USERNAME/BUT_Parking_Detection_CTM-and-LB.git
cd BUT_Parking_Detection_CTM-and-LB
```

### 2. Créer un environnement virtuel
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```powershell
pip install -r requirements.txt
```

### 4. Lancer le pipeline
```powershell
cd "modèle final"
python analyse_modele_final.py --help
```

## Structure du projet

```
.
├── DATA/                          # Dossier de données (à fournir)
├── modèle final/                  # Pipeline principal
│   ├── analyse_modele_final.py    # Analyse des images
│   ├── interface_selection_data.py # GUI launcher
│   ├── parking_zone.pkl           # Zones de travail
│   ├── zones_interdites.pkl       # Zones interdites
│   └── resultats_modele_final/    # Outputs images
├── requirements.txt               # Dépendances Python
└── README.md                      # Infos générales
```

## Dossiers importants à créer/remplir

- **`DATA/`** : Ajoute tes images par date (ex: `DATA/2026-03-25/`)
- **Modèles pré-entraînés** : `parking_detector_corrections.pt` et `yolov8m.pt` sont inclus dans le repo
- **Modèles YOLO supplémentaires** : Téléchargés automatiquement par YOLO si besoin

## Commandes principales

### Analyser les images d'une date
```powershell
cd "modèle final"
python analyse_modele_final.py --date 2026-03-25 --cleanup
```

### Configurer les zones de travail
```powershell
python config_zone_travail.py --image "path/to/image.jpg" --output "parking_zone.pkl" --profile-name "2026"
```

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'cv2'"**  
A: `pip install -r requirements.txt` n'a pas fonctionné. Réessaye.

**Q: "Permission denied" au lancement du venv**  
A: Lance PowerShell en tant qu'admin et réexécute le script d'activation.

## Contact / Support

En cas de problème, contacte moi ou vérifie les logs dans le fichier `.log`.

---
**Crée par:** [Ton nom]  
**Dernière mise à jour:** 16 avril 2026

#!/usr/bin/env python3
"""
Outil interactif pour la correction manuelle des détections de parking.
Interface Tkinter avec:
- Image cliquable à gauche
- Panel de contrôle à droite avec boutons et options
"""

import csv
import json
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import threading


DEFAULT_QUEUE_TXT = "manual_review_queue.txt"
DEFAULT_RESULTS_CSV = "resultats_modele_final.csv"
DEFAULT_REVIEWED_CSV = "manual_review_done.csv"
DEFAULT_ANNOTATIONS_JSON = "manual_review_annotations.json"
DEFAULT_OUTPUT_FOLDER = "resultats_modele_final"
DATA_BASE_DIR = Path(__file__).resolve().parent.parent / "DATA"
BASE_DIR = Path(__file__).resolve().parent


class InteractiveCorrectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Correction Manuelle - Interface Interactive")
        self.root.geometry("1400x800")

        # État
        self.images_queue = []
        self.current_idx = 0
        self.results_dict = {}
        self.reviewed_set = set()
        self.saved_states = {}
        
        # Corrections courantes
        self.cars_added = []  # [(x, y, type)] où type='legal' ou 'illegal'
        self.current_mode = None  # Mode sélectionné: 'legal' ou 'illegal'
        self.box_w_var = tk.IntVar(value=120)
        self.box_h_var = tk.IntVar(value=60)
        
        # Navigation
        self.next_button = None
        self.prev_button = None
        self.finish_button = None
        
        # État image
        self.image = None
        self.image_path = None
        self.image_display = None
        self.photo = None
        self.display_width = 0  # Largeur réelle affichée
        self.display_height = 0  # Hauteur réelle affichée
        
        # Valeurs originales
        self.original_total = 0
        self.original_forbidden = 0
        self.original_legal = 0

        # Build UI
        self._build_ui()
        
        # Charger données
        self.load_data_async()

    def _build_ui(self):
        """Construire l'interface"""
        # Frame principal horizontal
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # GAUCHE: Image cliquable
        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="Image (cliquer pour localiser les voitures)", 
                font=("Segoe UI", 10, "bold")).pack()
        
        self.image_label = tk.Label(left, bg="#e0e0e0", relief="solid", borderwidth=2)
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Button-1>", self.on_image_click)
        
        self.path_label = tk.Label(left, text="", font=("Consolas", 8), fg="#666")
        self.path_label.pack(fill="x", pady=(5, 0))

        # DROITE: Panel de contrôle SCROLLABLE
        right_container = tk.Frame(main)
        right_container.pack(side="right", fill="both", padx=(10, 0))
        
        # Canvas + scrollbar pour la partie contrôles
        canvas = tk.Canvas(right_container, bg="#f5f5f5", highlightthickness=0)
        scrollbar = tk.Scrollbar(right_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f5f5f5")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Alias pour simplifier
        right = scrollable_frame

        # Titre
        tk.Label(right, text="Contrôle", font=("Segoe UI", 12, "bold"), bg="#f5f5f5").pack(
            fill="x", padx=10, pady=(10, 10)
        )

        # Section: Valeurs originales
        tk.Label(right, text="Valeurs Originales", font=("Segoe UI", 10, "bold"), 
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(10, 5))
        
        orig_frame = tk.Frame(right, bg="#ffffff", relief="solid", borderwidth=1)
        orig_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.orig_total_var = tk.StringVar(value="0")
        self.orig_forbidden_var = tk.StringVar(value="0")
        self.orig_legal_var = tk.StringVar(value="0")
        
        tk.Label(orig_frame, text="Total:", bg="white").pack(anchor="w", padx=5, pady=3)
        tk.Label(orig_frame, textvariable=self.orig_total_var, font=("Consolas", 12, "bold"),
                bg="white", fg="#2196F3").pack(anchor="w", padx=10, pady=(0, 5))
        
        tk.Label(orig_frame, text="Illégales:", bg="white").pack(anchor="w", padx=5, pady=3)
        tk.Label(orig_frame, textvariable=self.orig_forbidden_var, font=("Consolas", 12, "bold"),
                bg="white", fg="#d63031").pack(anchor="w", padx=10, pady=(0, 5))
        
        tk.Label(orig_frame, text="Légales:", bg="white").pack(anchor="w", padx=5, pady=3)
        tk.Label(orig_frame, textvariable=self.orig_legal_var, font=("Consolas", 12, "bold"),
                bg="white", fg="#2e7d32").pack(anchor="w", padx=10, pady=(0, 5))

        # Separator
        tk.Frame(right, height=1, bg="#ccc").pack(fill="x", padx=10, pady=10)

        # Section: Choisir le type
        tk.Label(right, text="Action à effectuer:", font=("Segoe UI", 10, "bold"),
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(5, 10))
        
        type_frame = tk.Frame(right, bg="#f5f5f5")
        type_frame.pack(fill="x", padx=10, pady=(0, 15))
        
        self.mode_var = tk.StringVar(value=None)
        
        tk.Radiobutton(type_frame, text="Légale (vert) - ajouter", variable=self.mode_var,
                      value="legal", bg="#f5f5f5", font=("Segoe UI", 9),
                      command=self.on_mode_change).pack(anchor="w", pady=2)
        
        tk.Radiobutton(type_frame, text="Illégale (rouge) - ajouter", variable=self.mode_var,
                      value="illegal", bg="#f5f5f5", font=("Segoe UI", 9),
                      command=self.on_mode_change).pack(anchor="w", pady=2)
        
        tk.Radiobutton(type_frame, text="Non détectée (bleu) - faux négatif", variable=self.mode_var,
                      value="missed", bg="#f5f5f5", font=("Segoe UI", 9),
                      command=self.on_mode_change).pack(anchor="w", pady=2)
        
        tk.Radiobutton(type_frame, text="Faux positif (gris) - supprimer", variable=self.mode_var,
                      value="false_positive", bg="#f5f5f5", font=("Segoe UI", 9),
                      command=self.on_mode_change).pack(anchor="w", pady=2)

        # Instruction
        instr = tk.Label(right, text="Puis cliquez sur l'image", font=("Segoe UI", 9),
                        bg="#f5f5f5", fg="#666", justify="left")
        instr.pack(fill="x", padx=10, pady=(0, 15))

        # Taille de boite proposee
        tk.Label(right, text="Taille de boite proposée (px)", font=("Segoe UI", 10, "bold"),
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(0, 6))

        box_frame = tk.Frame(right, bg="#f5f5f5")
        box_frame.pack(fill="x", padx=10, pady=(0, 12))

        tk.Label(box_frame, text="Largeur", bg="#f5f5f5", anchor="w").pack(fill="x")
        tk.Scale(
            box_frame,
            from_=30,
            to=320,
            orient="horizontal",
            variable=self.box_w_var,
            resolution=2,
            bg="#f5f5f5",
            highlightthickness=0,
        ).pack(fill="x")

        tk.Label(box_frame, text="Hauteur", bg="#f5f5f5", anchor="w").pack(fill="x")
        tk.Scale(
            box_frame,
            from_=20,
            to=220,
            orient="horizontal",
            variable=self.box_h_var,
            resolution=2,
            bg="#f5f5f5",
            highlightthickness=0,
        ).pack(fill="x")

        # Separator
        tk.Frame(right, height=1, bg="#ccc").pack(fill="x", padx=10, pady=10)

        # Section: Actions
        tk.Label(right, text="Actions", font=("Segoe UI", 10, "bold"),
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(5, 10))
        
        tk.Button(right, text="Supprimer 1 voiture", command=self.remove_car,
                 bg="#ff9800", fg="white", font=("Segoe UI", 10), width=20).pack(
                     fill="x", padx=10, pady=3)
        
        tk.Button(right, text="Réinitialiser", command=self.reset,
                 bg="#9e9e9e", fg="white", font=("Segoe UI", 10), width=20).pack(
                     fill="x", padx=10, pady=3)

        # Separator
        tk.Frame(right, height=1, bg="#ccc").pack(fill="x", padx=10, pady=10)

        # Section: Valeurs corrigées
        tk.Label(right, text="Valeurs Corrigées", font=("Segoe UI", 10, "bold"),
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(5, 5))
        
        corr_frame = tk.Frame(right, bg="#fffacd", relief="solid", borderwidth=1)
        corr_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.corr_total_var = tk.StringVar(value="0")
        self.corr_forbidden_var = tk.StringVar(value="0")
        self.corr_legal_var = tk.StringVar(value="0")
        
        tk.Label(corr_frame, text="Total:", bg="#fffacd").pack(anchor="w", padx=5, pady=3)
        tk.Label(corr_frame, textvariable=self.corr_total_var, font=("Consolas", 12, "bold"),
                bg="#fffacd", fg="#2196F3").pack(anchor="w", padx=10, pady=(0, 5))
        
        tk.Label(corr_frame, text="Illégales:", bg="#fffacd").pack(anchor="w", padx=5, pady=3)
        tk.Label(corr_frame, textvariable=self.corr_forbidden_var, font=("Consolas", 12, "bold"),
                bg="#fffacd", fg="#d63031").pack(anchor="w", padx=10, pady=(0, 5))
        
        tk.Label(corr_frame, text="Légales:", bg="#fffacd").pack(anchor="w", padx=5, pady=3)
        tk.Label(corr_frame, textvariable=self.corr_legal_var, font=("Consolas", 12, "bold"),
                bg="#fffacd", fg="#2e7d32").pack(anchor="w", padx=10, pady=(0, 5))

        # Separator
        tk.Frame(right, height=1, bg="#ccc").pack(fill="x", padx=10, pady=10)

        # Compteur clics
        tk.Label(right, text="État", font=("Segoe UI", 10, "bold"),
                bg="#f5f5f5").pack(fill="x", padx=10, pady=(5, 5))
        
        self.state_label = tk.Label(right, text="", font=("Consolas", 9),
                                    bg="#f5f5f5", justify="left")
        self.state_label.pack(fill="x", padx=10, pady=(0, 10))

        # ================================================
        # BOUTONS EN BAS (NON-SCROLLABLE)
        # ================================================
        buttons_frame = tk.Frame(right_container, bg="#f5f5f5")
        buttons_frame.pack(fill="x", padx=10, pady=10)
        
        # Frame pour les boutons de navigation
        nav_frame = tk.Frame(buttons_frame, bg="#f5f5f5")
        nav_frame.pack(fill="x", pady=5)
        
        self.prev_button = tk.Button(nav_frame, text="← Précédente", command=self.prev_image,
                 bg="#FF9800", fg="white", font=("Segoe UI", 10), height=2)
        self.prev_button.pack(side="left", fill="both", expand=True, padx=(0, 2))
        
        self.next_button = tk.Button(nav_frame, text="Suivante →", command=self.next_image,
                 bg="#2196F3", fg="white", font=("Segoe UI", 10), height=2)
        self.next_button.pack(side="left", fill="both", expand=True, padx=(2, 0))
        
        # Bouton Finish (caché au début)
        self.finish_button = tk.Button(buttons_frame, text="TERMINER & QUITTER", command=self.finish_and_exit,
                 bg="#4CAF50", fg="white", font=("Segoe UI", 11, "bold"),
                 height=2)
        
        tk.Button(buttons_frame, text="Quitter", command=self.quit_app,
                 bg="#f44336", fg="white", font=("Segoe UI", 10), 
                 height=2).pack(fill="x", pady=5)

        # Status
        self.status_label = tk.Label(self.root, text="Initialisation...", 
                                    font=("Segoe UI", 9), fg="#666", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

    def on_mode_change(self):
        """Mode sélectionné"""
        mode = self.mode_var.get()
        if mode:
            self.current_mode = mode
            self.update_state()

    def on_image_click(self, event):
        """Clic sur l'image"""
        if not self.current_mode:
            messagebox.showwarning("Mode requis", "Choisir d'abord une action")
            return
        
        if self.image is None or self.display_width == 0 or self.display_height == 0:
            return
        
        # Obtenir taille du label
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        if label_width < 2 or label_height < 2:
            return
        
        # Calculer le décalage (image centrée dans le label)
        offset_x = (label_width - self.display_width) // 2
        offset_y = (label_height - self.display_height) // 2
        
        # Position du clic relative à l'image affichée
        x_on_display = event.x - offset_x
        y_on_display = event.y - offset_y
        
        # Vérifier qu'on a cliqué sur l'image
        if x_on_display < 0 or y_on_display < 0 or \
           x_on_display >= self.display_width or y_on_display >= self.display_height:
            return
        
        # Convertir vers coordonnées de l'image originale
        img_width, img_height = self.image.size
        
        scale_x = img_width / self.display_width if self.display_width > 0 else 1
        scale_y = img_height / self.display_height if self.display_height > 0 else 1
        
        x = int(x_on_display * scale_x)
        y = int(y_on_display * scale_y)
        
        # Clamper les coordonnées
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        # Ajouter le clic avec une boîte proposée
        self.cars_added.append((x, y, self.current_mode, int(self.box_w_var.get()), int(self.box_h_var.get())))
        self._store_current_state()
        
        # Redessiner
        self.display_image()
        self.update_state()

    def remove_car(self):
        """Supprimer une voiture"""
        if self.cars_added:
            self.cars_added.pop()
            self._store_current_state()
            self.display_image()
            self.update_state()

    def reset(self):
        """Réinitialiser les corrections"""
        self.cars_added = []
        self.current_mode = None
        self.mode_var.set(None)
        self._store_current_state()
        self.display_image()
        self.update_state()

    def _store_current_state(self):
        """Mémorise les corrections de l'image courante."""
        if self.image_path:
            self.saved_states[self.image_path] = list(self.cars_added)
            self._persist_annotations()

    def _load_saved_state(self, image_path: str):
        """Recharge les corrections déjà faites pour une image."""
        self.cars_added = list(self.saved_states.get(image_path, []))

    def _point_fields(self, point):
        """Normalise un point pour compatibilité ancien/nouveau format."""
        if isinstance(point, (list, tuple)):
            if len(point) >= 5:
                x, y, car_type, box_w, box_h = point[:5]
                return int(x), int(y), str(car_type), max(0, int(box_w)), max(0, int(box_h))
            if len(point) >= 3:
                x, y, car_type = point[:3]
                return int(x), int(y), str(car_type), 0, 0
        return 0, 0, "legal", 0, 0

    def _persist_annotations(self):
        """Écrit les corrections en JSON pour réutilisation à l'entraînement."""
        annotations_path = BASE_DIR / DEFAULT_ANNOTATIONS_JSON
        payload = {
            "schema_version": 1,
            "images": {
                image_key: [
                    {
                        "x": x,
                        "y": y,
                        "type": car_type,
                        "box_w": box_w,
                        "box_h": box_h,
                    }
                    for x, y, car_type, box_w, box_h in (self._point_fields(p) for p in points)
                ]
                for image_key, points in self.saved_states.items()
            },
        }
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update_state(self):
        """Met à jour l'affichage des valeurs corrigées"""
        legal = self.original_legal
        illegal = self.original_forbidden
        
        # Applique les corrections
        for x, y, car_type, _, _ in (self._point_fields(p) for p in self.cars_added):
            if car_type == "legal":
                legal += 1
            elif car_type == "illegal":
                illegal += 1
            elif car_type == "missed":
                # Voiture non détectée = on ajoute une voiture légale
                legal += 1
            elif car_type == "false_positive":
                # Faux positif = on soustrait du total (pas de type spécifique)
                # On soustrait d'abord des illégales, puis des légales
                if illegal > 0:
                    illegal -= 1
                elif legal > 0:
                    legal -= 1
        
        total = legal + illegal
        
        self.corr_total_var.set(str(total))
        self.corr_forbidden_var.set(str(illegal))
        self.corr_legal_var.set(str(legal))
        
        state_text = f"Clics en mémoire: {len(self.cars_added)}"
        self.state_label.config(text=state_text)

    def display_image(self):
        """Affiche l'image avec les points"""
        if self.image is None:
            return
        
        # Créer une copie pour l'affichage
        display = self.image.copy()
        draw = ImageDraw.Draw(display)
        
        # Dessiner les boites/points corriges
        for x, y, car_type, box_w, box_h in (self._point_fields(p) for p in self.cars_added):
            if car_type == "legal":
                color = (0, 200, 0)  # Vert
            elif car_type == "illegal":
                color = (255, 0, 0)  # Rouge
            elif car_type == "missed":
                color = (0, 0, 255)  # Bleu (voiture non détectée)
            else:  # false_positive
                color = (128, 128, 128)  # Gris (faux positif)

            bw = box_w if box_w > 0 else 120
            bh = box_h if box_h > 0 else 60
            x1 = max(0, x - bw // 2)
            y1 = max(0, y - bh // 2)
            x2 = min(display.width - 1, x + bw // 2)
            y2 = min(display.height - 1, y + bh // 2)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        
        # Redimensionner pour fit
        display.thumbnail((700, 750), Image.Resampling.LANCZOS)
        
        # Mémoriser les dimensions réelles affichées
        self.display_width, self.display_height = display.size
        
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.config(image=self.photo)

    def load_data_async(self):
        """Charger données en background"""
        thread = threading.Thread(target=self._load_data, daemon=True)
        thread.start()

    def _processed_image_path(self, image_relpath: str) -> Path:
        """Retourne l'image annotée par YOLO si elle existe, sinon l'image brute."""
        relpath = image_relpath.replace("\\", "/")
        safe_rel = relpath.replace("/", "__").replace(":", "_")

        annotated_path = BASE_DIR / DEFAULT_OUTPUT_FOLDER / f"ModeleFinal_{safe_rel}"
        if annotated_path.exists():
            return annotated_path

        legacy_path = BASE_DIR / DEFAULT_OUTPUT_FOLDER / f"ModeleFinal_{Path(relpath).name}"
        if legacy_path.exists():
            return legacy_path

        return DATA_BASE_DIR / image_relpath

    def _load_data(self):
        """Charge queue et CSV"""
        queue_path = BASE_DIR / DEFAULT_QUEUE_TXT
        if not queue_path.exists():
            self.root.after(0, lambda: messagebox.showerror(
                "Erreur",
                f"{DEFAULT_QUEUE_TXT} introuvable.\nLancer d'abord: preparer_lot_correction.py"
            ))
            return

        with open(queue_path, "r", encoding="utf-8") as f:
            self.images_queue = [line.strip() for line in f if line.strip()]

        if not self.images_queue:
            self.root.after(0, lambda: messagebox.showinfo("Queue Vide", "Aucune image à corriger."))
            return

        # Charger CSV
        csv_path = BASE_DIR / DEFAULT_RESULTS_CSV
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    image = row.get("image", "").strip()
                    if image:
                        self.results_dict[image] = row

        # Charger déjà révisées
        reviewed_path = BASE_DIR / DEFAULT_REVIEWED_CSV
        if reviewed_path.exists():
            with open(reviewed_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    image = row.get("image", "").strip()
                    if image:
                        self.reviewed_set.add(image)

        annotations_path = BASE_DIR / DEFAULT_ANNOTATIONS_JSON
        if annotations_path.exists():
            try:
                with open(annotations_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    images = payload.get("images", {})
                    if isinstance(images, dict):
                        for image_key, points in images.items():
                            if not isinstance(points, list):
                                continue
                            restored = []
                            for item in points:
                                if isinstance(item, dict):
                                    restored.append(
                                        (
                                            int(item.get("x", 0)),
                                            int(item.get("y", 0)),
                                            str(item.get("type", "legal")),
                                            int(item.get("box_w", 0) or 0),
                                            int(item.get("box_h", 0) or 0),
                                        )
                                    )
                            if restored:
                                self.saved_states[image_key] = restored
            except Exception:
                pass

        self.current_idx = 0
        self.root.after(0, self.show_image)

    def show_image(self):
        """Affiche l'image courante"""
        if self.current_idx >= len(self.images_queue):
            # Fin des images - afficher le bouton Finish
            self.prev_button.pack_forget()
            self.next_button.pack_forget()
            self.finish_button.pack(fill="x", pady=5)
            
            self.image_label.config(image="")
            self.image = None
            self.path_label.config(text="✅ Toutes les images ont été traitées!")
            self.status_label.config(text="Cliquez sur TERMINER & QUITTER pour finir", fg="#4CAF50")
            return

        # Montrer les boutons standards
        self.finish_button.pack_forget()
        self.prev_button.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.next_button.pack(side="left", fill="both", expand=True, padx=(2, 0))
        
        # Activer/désactiver le bouton Précédente
        if self.current_idx > 0:
            self.prev_button.config(state="normal")
        else:
            self.prev_button.config(state="disabled")

        # Mettre le label "Suivante" ou "SAUVEGARDER" selon position
        self.next_button.config(text="Suivante →")

        self.image_path = self.images_queue[self.current_idx]
        full_path = self._processed_image_path(self.image_path)

        if not full_path.exists():
            self.status_label.config(text=f"⚠ Image introuvable: {full_path}")
            self.current_idx += 1
            self.show_image()
            return

        try:
            self.image = Image.open(full_path)
        except Exception as e:
            self.status_label.config(text=f"⚠ Erreur: {e}")
            self.current_idx += 1
            self.show_image()
            return

        # Charger valeurs originales
        if self.image_path in self.results_dict:
            row = self.results_dict[self.image_path]
            self.original_total = int(row.get("total_cars", "0"))
            self.original_forbidden = int(row.get("cars_in_forbidden", "0"))
            self.original_legal = int(row.get("cars_legal", "0"))
        else:
            self.original_total = 0
            self.original_forbidden = 0
            self.original_legal = 0

        self.orig_total_var.set(str(self.original_total))
        self.orig_forbidden_var.set(str(self.original_forbidden))
        self.orig_legal_var.set(str(self.original_legal))

        # Restore corrections for this image if we already visited it
        self.current_mode = None
        self.mode_var.set(None)
        self._load_saved_state(self.image_path)

        # Afficher
        self.path_label.config(text=f"{self.image_path}  |  {full_path.name}")
        progress = f"Image {self.current_idx + 1}/{len(self.images_queue)}"
        self.status_label.config(text=progress, fg="#666")
        
        self.display_image()
        self.update_state()

    def save_and_next(self):
        """Enregistre l'état courant en mémoire et continue."""
        self._store_current_state()
        self.current_idx += 1
        self.show_image()

    def next_image(self):
        """Suivant avec conservation en mémoire."""
        self._store_current_state()
        self.current_idx += 1
        self.show_image()

    def prev_image(self):
        """Revenir à l'image précédente"""
        self._store_current_state()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_image()

    def finish_and_exit(self):
        """Terminer et quitter"""
        self._store_current_state()
        self._persist_annotations()
        self._save_all_corrections()
        messagebox.showinfo("Bravo!", "Correction manuelle terminée avec succès!")
        self.root.quit()

    def _save_all_corrections(self):
        """Sauvegarde toutes les corrections dans le CSV et le journal de revue."""
        csv_path = BASE_DIR / DEFAULT_RESULTS_CSV
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                image_key = row.get("image", "").strip()
                saved_points = self.saved_states.get(image_key)
                if saved_points is not None:
                    legal = int(row.get("cars_legal", "0") or 0)
                    illegal = int(row.get("cars_in_forbidden", "0") or 0)

                    for _, _, car_type, _, _ in (self._point_fields(p) for p in saved_points):
                        if car_type == "legal":
                            legal += 1
                        elif car_type == "illegal":
                            illegal += 1
                        elif car_type == "missed":
                            legal += 1
                        elif car_type == "false_positive":
                            if illegal > 0:
                                illegal -= 1
                            elif legal > 0:
                                legal -= 1

                    row["total_cars"] = str(legal + illegal)
                    row["cars_in_forbidden"] = str(illegal)
                    row["cars_legal"] = str(legal)
                rows.append(row)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        reviewed_path = BASE_DIR / DEFAULT_REVIEWED_CSV
        reviewed_rows = []
        if reviewed_path.exists():
            with open(reviewed_path, "r", newline="", encoding="utf-8") as f:
                reviewed_rows = list(csv.DictReader(f))
        
        reviewed_map = {r.get("image", "").strip(): r for r in reviewed_rows if r.get("image", "").strip()}
        for image_key, saved_points in self.saved_states.items():
            source_row = self.results_dict.get(image_key, {})
            legal = int(source_row.get("cars_legal", "0") or 0)
            illegal = int(source_row.get("cars_in_forbidden", "0") or 0)

            for _, _, car_type, _, _ in (self._point_fields(p) for p in saved_points):
                if car_type == "legal":
                    legal += 1
                elif car_type == "illegal":
                    illegal += 1
                elif car_type == "missed":
                    legal += 1
                elif car_type == "false_positive":
                    if illegal > 0:
                        illegal -= 1
                    elif legal > 0:
                        legal -= 1

            reviewed_map[image_key] = {
                "image": image_key,
                "corrected_total": str(legal + illegal),
                "corrected_forbidden": str(illegal),
                "corrected_legal": str(legal),
            }
        
        with open(reviewed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "corrected_total", "corrected_forbidden", "corrected_legal"])
            writer.writeheader()
            writer.writerows(reviewed_map.values())

        self.reviewed_set.update(self.saved_states.keys())

    def quit_app(self):
        """Quitter"""
        if messagebox.askokcancel("Quitter", "Quitter la correction ?"):
            self.root.quit()


def main():
    root = tk.Tk()
    app = InteractiveCorrectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

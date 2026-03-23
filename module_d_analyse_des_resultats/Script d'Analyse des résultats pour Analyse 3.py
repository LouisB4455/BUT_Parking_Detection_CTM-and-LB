import os
import csv
from tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageFont

# CONFIG
IMAGE_FOLDER = "Images - Analyse 3"
OUTPUT_FOLDER = "Analyse 3 - VF"
CSV_FILE = "Analyse 3 - VF.csv"
TOTAL_PLACES = 37

ERROR_CODES = {
    1: "Voiture non détectée",
    2: "Fausse détection",
    3: "Stationnement sauvage",
    4: "Voiture partielle",
    5: "Image inexploitable",
    6: "Obstacle non voiture",
    7: "Place non visible",
    8: "Voiture sur 2 places",
    9: "Double détection"
}

class ParkingAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Parking")
        self.root.state('zoomed')

        self.images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.index = 0

        self.original_image = None
        self.current_image = None
        self.display_image_pil = None
        self.tk_image = None

        self.scale_ratio = 1

        self.annotations = []
        self.current_error = None

        try:
            self.font = ImageFont.truetype("arial.ttf", 40)
        except:
            self.font = ImageFont.load_default()

        self.setup_ui()
        self.bind_keys()
        self.load_image()

    def setup_ui(self):
        self.left_frame = Frame(self.root, width=320, bg="#2c2c2c")
        self.left_frame.pack(side=LEFT, fill=Y)

        self.right_frame = Frame(self.root)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # 🔥 MESSAGE IMPORTANT
        self.warning_label = Label(
            self.right_frame,
            text="⚠️ Pense à signaler les véhicules en stationnement sauvage",
            bg="yellow",
            fg="black",
            font=("Arial", 14, "bold")
        )
        self.warning_label.pack(fill=X)

        self.canvas = Canvas(self.right_frame, bg="black")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        Label(self.left_frame, text="Codes erreurs", bg="#2c2c2c", fg="white", font=("Arial", 14)).pack(pady=10)

        for code, text in ERROR_CODES.items():
            Button(
                self.left_frame,
                text=f"{code}. {text}",
                command=lambda c=code: self.select_error(c),
                wraplength=250,
                justify=LEFT
            ).pack(fill=X, padx=10, pady=3)

        Button(self.left_frame, text="Suivant", bg="green", fg="white", command=self.next_image)\
            .pack(pady=10, padx=10, fill=X)

        self.preview_label = Label(
            self.left_frame,
            text="Prévisualisation CSV",
            bg="#2c2c2c",
            fg="white",
            wraplength=280,
            justify=LEFT
        )
        self.preview_label.pack(pady=20, padx=10)

    def bind_keys(self):
        self.root.bind("e", self.reset_annotations)
        self.root.bind("E", self.reset_annotations)

    def load_image(self):
        if self.index >= len(self.images):
            print("✅ Terminé !")
            return

        path = os.path.join(IMAGE_FOLDER, self.images[self.index])
        self.original_image = Image.open(path).convert("RGB")
        self.current_image = self.original_image.copy()

        self.annotations = []
        self.display_image()
        self.update_preview()

    def display_image(self):
        img = self.current_image.copy()

        screen_w = self.root.winfo_screenwidth() - 320
        screen_h = self.root.winfo_screenheight()

        w, h = img.size
        self.scale_ratio = min(screen_w / w, screen_h / h)

        new_size = (int(w * self.scale_ratio), int(h * self.scale_ratio))
        self.display_image_pil = img.resize(new_size)

        self.tk_image = ImageTk.PhotoImage(self.display_image_pil)

        self.canvas.delete("all")
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def select_error(self, code):
        self.current_error = code

    def on_click(self, event):
        if self.current_error is None:
            return

        x, y = event.x, event.y
        real_x = int(x / self.scale_ratio)
        real_y = int(y / self.scale_ratio)

        r = 15

        self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="red", width=2)
        self.canvas.create_rectangle(x-14, y-14, x+14, y+14, fill="white")
        self.canvas.create_text(x, y, text=str(self.current_error), fill="black", font=("Arial", 14, "bold"))

        draw = ImageDraw.Draw(self.current_image)

        text = str(self.current_error)
        bbox = self.font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 10

        draw.rectangle(
            (
                real_x - text_w//2 - padding,
                real_y - text_h//2 - padding,
                real_x + text_w//2 + padding,
                real_y + text_h//2 + padding
            ),
            fill="white"
        )

        draw.ellipse((real_x-r, real_y-r, real_x+r, real_y+r), outline="red", width=3)

        draw.text(
            (real_x - text_w//2, real_y - text_h//2),
            text,
            fill="black",
            font=self.font
        )

        self.annotations.append({
            "error": self.current_error,
            "x": real_x,
            "y": real_y
        })

        self.update_preview()

    def update_preview(self):
        error_counts = {code: 0 for code in ERROR_CODES}
        for ann in self.annotations:
            error_counts[ann["error"]] += 1

        detected_places = TOTAL_PLACES - error_counts[1]

        coords_preview = " | ".join([f"{a['error']}:{a['x']}:{a['y']}" for a in self.annotations])

        text = f"Image : {self.images[self.index]}\n\n"
        for code in ERROR_CODES:
            text += f"{ERROR_CODES[code]} : {error_counts[code]}\n"

        text += f"\nPlaces détectées : {detected_places}\n\n"
        text += f"Coords : {coords_preview}"

        self.preview_label.config(text=text)

    def reset_annotations(self, event=None):
        self.current_image = self.original_image.copy()
        self.annotations = []
        self.display_image()
        self.update_preview()

    def next_image(self):
        self.save_results()
        self.index += 1
        self.load_image()

    def save_results(self):
        filename = self.images[self.index]

        error_counts = {code: 0 for code in ERROR_CODES}
        for ann in self.annotations:
            error_counts[ann["error"]] += 1

        detected_places = TOTAL_PLACES - error_counts[1]

        coords = " | ".join([f"{a['error']}:{a['x']}:{a['y']}" for a in self.annotations])

        file_exists = os.path.isfile(CSV_FILE)

        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                header = ["image"] + [f"err{code}" for code in ERROR_CODES] + ["places_detectees", "coords"]
                writer.writerow(header)

            row = [filename] + [error_counts[code] for code in ERROR_CODES] + [detected_places, coords]
            writer.writerow(row)

        output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
        self.current_image.save(output_path)

        print(f"💾 Sauvegardé : {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    root = Tk()
    app = ParkingAnnotator(root)
    root.mainloop()

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../DATA").resolve()


class DataSelectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modele Final - Selection Dossiers DATA")
        self.root.geometry("900x620")

        title = tk.Label(
            root,
            text="Selection des sous-dossiers DATA a traiter",
            font=("Segoe UI", 13, "bold"),
        )
        title.pack(pady=(12, 6))

        self.path_label = tk.Label(
            root,
            text=f"DATA: {DATA_DIR}",
            anchor="w",
            justify="left",
            font=("Segoe UI", 9),
        )
        self.path_label.pack(fill="x", padx=12)

        frame_list = tk.Frame(root)
        frame_list.pack(fill="both", expand=True, padx=12, pady=8)

        self.listbox = tk.Listbox(
            frame_list,
            selectmode=tk.EXTENDED,
            font=("Consolas", 10),
            activestyle="dotbox",
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        frame_buttons = tk.Frame(root)
        frame_buttons.pack(fill="x", padx=12, pady=6)

        tk.Button(frame_buttons, text="Rafraichir", command=self.refresh_folders).pack(side="left")
        tk.Button(frame_buttons, text="Tout selectionner", command=self.select_all).pack(side="left", padx=6)
        tk.Button(frame_buttons, text="Tout deselectionner", command=self.clear_selection).pack(side="left")

        self.run_button = tk.Button(
            frame_buttons,
            text="Lancer Modele Sur Selection",
            command=self.run_pipeline,
            bg="#1f6feb",
            fg="white",
        )
        self.run_button.pack(side="right")

        self.open_html_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            root,
            text="Ouvrir monitoring_final_simple.html a la fin",
            variable=self.open_html_var,
        ).pack(anchor="w", padx=14)

        tk.Label(root, text="Journal d'execution:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=12, pady=(8, 2)
        )

        self.log_text = tk.Text(root, height=14, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=False, padx=12, pady=(0, 12))

        self.refresh_folders()

    def log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def refresh_folders(self) -> None:
        self.listbox.delete(0, tk.END)
        if not DATA_DIR.exists():
            self.log(f"[ERREUR] Dossier DATA introuvable: {DATA_DIR}")
            return

        folders = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
        for name in folders:
            self.listbox.insert(tk.END, name)

        self.log(f"{len(folders)} sous-dossiers detectes dans DATA")

    def select_all(self) -> None:
        self.listbox.select_set(0, tk.END)

    def clear_selection(self) -> None:
        self.listbox.selection_clear(0, tk.END)

    def get_selection(self) -> list[str]:
        indices = self.listbox.curselection()
        return [self.listbox.get(i) for i in indices]

    def run_pipeline(self) -> None:
        selected = self.get_selection()
        if not selected:
            messagebox.showwarning("Selection vide", "Selectionne au moins un sous-dossier DATA.")
            return

        self.run_button.config(state="disabled")
        thread = threading.Thread(target=self._run_pipeline_thread, args=(selected,), daemon=True)
        thread.start()

    def _run_command(self, cmd: list[str]) -> int:
        self.log("\n$ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log(line.rstrip())
        return proc.wait()

    def _run_pipeline_thread(self, selected: list[str]) -> None:
        try:
            py = sys.executable

            self.log("[INFO] Entrainement batch decouple: utiliser run_batch_training.bat quand souhaite.")

            self.log("\n=== Lancement modele final sur sous-dossiers selectionnes ===")
            cmd = [
                py,
                "analyse_modele_final.py",
                "--input-folder",
                "../DATA",
                "--include-subfolders",
                *selected,
                "--model-path",
                "parking_detector_corrections.pt",
                "--forbidden-overlap-threshold",
                "0.25",
                "--forbidden-min-overlap-pixels",
                "40",
                "--forbidden-zones",
                "zones_interdites.pkl",
                "--work-zone",
                "parking_zone.pkl",
            ]
            code = self._run_command(cmd)
            if code != 0:
                self.log(f"[ERREUR] analyse_modele_final.py a retourne {code}")
                return

            self.log("\n=== Mise a jour monitoring HTML ===")
            code = self._run_command([py, "mettre_a_jour_monitoring_html.py"])
            if code != 0:
                self.log(f"[ERREUR] mettre_a_jour_monitoring_html.py a retourne {code}")
                return

            self.log("\n[OK] Pipeline termine.")

            if self.open_html_var.get():
                html_path = BASE_DIR / "monitoring_final_simple.html"
                if html_path.exists():
                    os.startfile(str(html_path))
        finally:
            self.run_button.config(state="normal")


def main() -> None:
    root = tk.Tk()
    app = DataSelectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

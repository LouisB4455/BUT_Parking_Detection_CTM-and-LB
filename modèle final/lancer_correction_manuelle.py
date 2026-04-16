import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    correction_gui = script_dir / "correction_gui.py"

    print("Lancement de l'outil de correction manuelle GUI...")
    print(f"Dossier: {script_dir}")

    # Lancer correction_gui.py
    result = subprocess.run(
        [sys.executable, str(correction_gui)],
        cwd=str(script_dir),
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

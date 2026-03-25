import glob
import os
import runpy


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.normpath(os.path.join(current_dir, "..", "module_d_analyse_des_resultats"))

    pattern = os.path.join(module_dir, "*Analyse 1.py")
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"Script de correction manuelle introuvable via: {pattern}")

    script_path = matches[0]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()

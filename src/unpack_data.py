import argparse
from pathlib import Path
import pandas as pd

def unpack_data(input_dir: str, output_file: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_file)

    all_files = []
    
    # On parcourt récursivement
    for f in input_path.rglob("*"):
        if f.is_file():
            # On calcule la distance entre le fichier et le dossier racine
            # archive/dev/file -> depth 1
            # archive/random_split/dev/file -> depth 2
            # archive/random_split/random_split/dev/file -> depth 3 (TROP LOIN)
            depth = len(f.relative_to(input_path).parts)

            # FILTRES :
            # 1. On ignore le dossier 'output'
            # 2. On ignore les fichiers trop profonds (depth > 2 ou 3 selon ta structure)
            if "output" not in f.parts and depth <= 3:
                all_files.append(f)

    if not all_files:
        print(f" Aucun fichier source trouvé dans la limite de profondeur autorisée.")
        return

    print(f" {len(all_files)} fichiers identifiés (profondeur limitée).")

    dataframes = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f" Erreur lecture {file_path.name} : {e}")

    if not dataframes:
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"--- Terminé ---")
    print(f" Fichier final : {output_path}")
    print(f" Total lignes : {len(combined_df)}")

if __name__ == "__main__":
    unpack_data(
        input_dir=r"data\bronze\archive", 
        output_file=r"data\bronze\archive\output\output_file.csv"
    )
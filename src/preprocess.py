import argparse
from pathlib import Path

import pandas as pd


def preprocess_data(data_file: str, output_dir: str) -> None:
    """
    Preprocess raw protein sequence data for model training.
    """

    data_path = Path(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the data
    df = pd.read_csv(data_path)

    # 2. Remove rows with missing values
    df = df.dropna()

    # 3. Encode labels (pandas equivalent of LabelEncoder)
    df["family_encoded"], _ = pd.factorize(df["family_accession"])

    train_parts = []
    val_parts = []
    test_parts = []

    # 4. Custom split strategy inspired by data analysis notebook
    for _, group in df.groupby("family_encoded"):
        n = len(group)
        group = group.sample(frac=1, random_state=42)  # shuffle

        if n == 1:
            # Only train
            train_parts.append(group)

        elif n == 2:
            # Train + validation
            train_parts.append(group.iloc[:1])
            val_parts.append(group.iloc[1:2])

        else:
            # Train / Val / Test
            train_size = int(0.7 * n)
            val_size = int(0.15 * n)

            train_parts.append(group.iloc[:train_size])
            val_parts.append(group.iloc[train_size:train_size + val_size])
            test_parts.append(group.iloc[train_size + val_size:])

    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts) if test_parts else pd.DataFrame(columns=df.columns)

    # 5. Save processed datasets
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    preprocess_data(
        data_file=r"data\bronze\archive\output\output_file.csv", 
        output_dir=r"data\silver"
    )

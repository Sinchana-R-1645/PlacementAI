
import pandas as pd
from pathlib import Path

# Location of the datasets
data_folder = Path(__file__).parent / "data"

datasets = {
    "Placement Prediction": "placement_dataset.csv",
    "Job Recommendation": "job_roles.csv"
}

for name, filename in datasets.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    file_path = data_folder / filename

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)

        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nMissing values in each column:")
        print(df.isnull().sum().to_string())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as error:
        print("Could not read the file:", error)

print("\nDataset checking completed.")

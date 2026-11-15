
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load the dataset
filename = "Diabetes_Final_Data_V2.csv"

data_path = Path("data") / filename
if not data_path.exists():
    data_path = Path(filename)

df = pd.read_csv(data_path)

print("Original dataset:", df.shape)

# 2. Split into 70% training and 30% remaining
train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["diabetic"]
)

# 3. Split remaining 30% into 20% test and 10% validation
test_df, validation_df = train_test_split(
    remaining_df,
    test_size=1/3,
    random_state=42,
    stratify=remaining_df["diabetic"]
)

# 4. Save datasets as separate CSV files
output_dir = Path("data") / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)
validation_df.to_csv(
    output_dir / "validation_unseen.csv",
    index=False
)

# 5. Display results
print("\nDataset Split Results:")
print("Training:", train_df.shape)
print("Testing:", test_df.shape)
print("Validation (Unseen):", validation_df.shape)

print("\nFiles saved successfully!")

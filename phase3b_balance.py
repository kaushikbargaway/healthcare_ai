import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("data/processed/severity_dataset.csv")

low = df[df.severity == 0]
mod = df[df.severity == 1]
high = df[df.severity == 2]

# Upsample low and high to match moderate
low_up = resample(low, replace=True, n_samples=len(mod), random_state=42)
high_up = resample(high, replace=True, n_samples=len(mod), random_state=42)

balanced = pd.concat([low_up, mod, high_up])
balanced = balanced.sample(frac=1).reset_index(drop=True)

balanced.to_csv(
    "data/processed/severity_dataset_balanced.csv",
    index=False
)

print("Balanced class counts:")
print(balanced.severity.value_counts())

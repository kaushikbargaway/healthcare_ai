import pandas as pd

df = pd.read_csv("data/processed/severity_dataset_balanced.csv")

texts = df["clean_text"].tolist()
labels = df["severity"].tolist()

print("Samples:", len(texts))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print(embeddings.shape)

import numpy as np

np.save("data/processed/X_embeddings.npy", embeddings)
np.save("data/processed/y_labels.npy", labels)

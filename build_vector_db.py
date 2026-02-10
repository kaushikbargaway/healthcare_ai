import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load chunks
chunks = []
with open("data/processed/chunks.txt", encoding="utf-8") as f:
    for line in f:
        chunks.append(line.strip())

print("Chunks loaded:", len(chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to vectors
embeddings = model.encode(chunks, show_progress_bar=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "vector_db/medical_index.faiss")

# Save chunks mapping
np.save("vector_db/chunks.npy", np.array(chunks))

print("Vector database created successfully!")
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "chest pain and breathing difficulty"
q_vec = model.encode([query])

D, I = index.search(q_vec, k=3)

print("Top Matches:\n")
for idx in I[0]:
    print(chunks[idx][:200])
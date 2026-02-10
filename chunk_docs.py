import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

src = "data/medical_docs"
out = "data/processed/chunks.txt"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

all_chunks = []

for file in os.listdir(src):
    if file.endswith(".txt"):
        text = open(os.path.join(src, file), encoding="utf-8").read()
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

print("Total Chunks:", len(all_chunks))

with open(out, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(c.replace("\n", " ") + "\n")
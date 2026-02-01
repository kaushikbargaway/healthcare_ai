from bs4 import BeautifulSoup
import os

src = "data/medical_docs_html"
dst = "data/medical_docs"

os.makedirs(dst, exist_ok=True)

for file in os.listdir(src):
    if file.endswith(".html"):   # ✅ only HTML files
        path = os.path.join(src, file)
        html = open(path, encoding="utf-8").read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        out_file = file.replace(".html", ".txt")
        open(os.path.join(dst, out_file), "w", encoding="utf-8").write(text)

print("Conversion completed successfully!")
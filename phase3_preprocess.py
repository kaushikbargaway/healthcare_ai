import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

df = pd.read_csv("data/raw/symptom_disease.csv")

# Remove useless column
df = df.drop(columns=["Unnamed: 0"])

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

severity_map = {
 "Heart attack":2,
 "Stroke":2,
 "Hypertension":2,
 "Diabetes":1,
 "Pneumonia":2,
 "Migraine":1,
 "Bronchial Asthma":1,
 "Asthma":1,
 "Typhoid":1,
 "Psoriasis":0,
 "Common Cold":0,
 "Allergy":0,
 "Acne":0
}

def map_severity(disease):
    return severity_map.get(disease,1)  # default moderate

df["severity"] = df["label"].apply(map_severity)

final_df = df[["clean_text","severity"]]

final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv(
 "data/processed/severity_dataset.csv",
 index=False
)

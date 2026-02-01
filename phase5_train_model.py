import numpy as np

X = np.load("data/processed/X_embeddings.npy")
y = np.load("data/processed/y_labels.npy")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

import joblib
joblib.dump(model, "models/severity_model.pkl")

loaded = joblib.load("models/severity_model.pkl")
print(loaded.predict([X_test[0]]))

import numpy as np
print("Class distribution:", np.bincount(y))
# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ===============================
# 2. Create Fake Maintenance Logs Dataset
# ===============================

logs = [
    "Heavy dust accumulation observed on panels",
    "Dust layer reducing panel efficiency",
    "Panels require cleaning due to dirt",
    "Surface contamination detected",
    
    "Panel temperature exceeded 85 degrees",
    "Module overheating detected",
    "Abnormally high temperature recorded",
    "Thermal sensor shows overheating",
    
    "Inverter output fluctuating frequently",
    "Inverter shutdown unexpectedly",
    "Power conversion instability observed",
    "Voltage irregularity in inverter",
    
    "System operating normally",
    "No issues detected",
    "Routine inspection completed successfully",
    "All parameters within normal range"
]

labels = [
    "Panel Cleaning Required",
    "Panel Cleaning Required",
    "Panel Cleaning Required",
    "Panel Cleaning Required",
    
    "Module Overheating",
    "Module Overheating",
    "Module Overheating",
    "Module Overheating",
    
    "Inverter Issue",
    "Inverter Issue",
    "Inverter Issue",
    "Inverter Issue",
    
    "No Issue",
    "No Issue",
    "No Issue",
    "No Issue"
]

df = pd.DataFrame({
    "log_text": logs,
    "label": labels
})

# ===============================
# 3. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# 4. TF-IDF Vectorization
# ===============================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 5. Train Classifier
# ===============================

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ===============================
# 6. Evaluation
# ===============================

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save Model + Vectorizer
# ===============================

pickle.dump(model, open("maintenance_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and Vectorizer Saved Successfully!")

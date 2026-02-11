import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("maintenance_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Batch Maintenance Log Classifier")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert logs into vectors
    X_vec = vectorizer.transform(df["log_text"])

    # Predict
    df["Predicted_Issue"] = model.predict(X_vec)

    st.dataframe(df)

import pickle

model = pickle.load(open("maintenance_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

new_log = ["Temperature exceeded safe threshold"]

vec = vectorizer.transform(new_log)
prediction = model.predict(vec)

print("Predicted Issue:", prediction[0])

##Developed an NLP-based multi-class maintenance log classifier using TF-IDF and Logistic Regression to detect operational issues in solar power plants.
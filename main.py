import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib

DATASET_FILENAME = "spam_data.csv"

def load_data():
    if os.path.exists(DATASET_FILENAME):
        try:
            df = pd.read_csv(DATASET_FILENAME, encoding="latin-1", engine="python")
            
            if {'v1','v2'}.issubset(df.columns):
                df = df[['v1','v2']].rename(columns={'v1':'Label','v2':'Message'})
            elif {'label','text'}.issubset(df.columns):
                df = df[['label','text']].rename(columns={'label':'Label','text':'Message'})
            elif {'Label','Message'}.issubset(df.columns):
                df = df[['Label','Message']]
            else:
                
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    df = df[[cols[0], cols[1]]].rename(columns={cols[0]:'Label', cols[1]:'Message'})
                else:
                    raise ValueError("CSV doesn't have expected columns")
            df = df.dropna(subset=['Label','Message'])
            
            df['Label'] = df['Label'].astype(str).str.strip().str.lower()
            return df
        except Exception as e:
            print("Error reading file, using sample data instead:", e)

    data = {
        "Label": ["ham","spam","ham","spam"],
        "Message": [
            "Hi are you coming today?",
            "You won free tickets!! call now",
            "I'll reach in 10 mins",
            "Claim your prize by sending code"
        ]
    }
    return pd.DataFrame(data)

def preprocess(df):
    df = df.dropna(subset=['Message','Label'])
    df['Spam'] = df['Label'].map(lambda x: 1 if str(x).strip().lower()=='spam' else 0)

    X = df['Message']
    y = df['Spam'].astype(int)

    strat = y if len(y.unique())>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def train_nb(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

def evaluate(model, X_test_vec, y_test):
    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print()
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def predict_one(model, vectorizer, text):
    vec = vectorizer.transform([text])
    p = model.predict(vec)[0]
    label = "SPAM" if p==1 else "HAM"

    conf = None
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(vec)[0].max()
        print(f"{label} (confidence: {conf:.2f})")
    else:
        print(label)

if __name__ == "__main__":
    df = load_data()
    X_train_vec, X_test_vec, y_train, y_test, vec = preprocess(df)
    model = train_nb(X_train_vec, y_train)
    evaluate(model, X_test_vec, y_test)

    print("\n--- Testing some messages ---")
    predict_one(model, vec, "Congratulations! You won a free phone!")
    predict_one(model, vec, "Hey bro, what's up?")

    joblib.dump(model, "spam_nb_model.joblib")
    joblib.dump(vec, "tfidf_vectorizer.joblib")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import io

# --- 1. DATA PREPARATION AND LOADING (Module 1 Component) ---

DATASET_FILENAME = 'spam_data.csv'

def load_data():
    """
    Loads the dataset from a CSV file. If the file is not found,
    it creates a small mock dataset for demonstration.
    
    In a real project, replace the mock data with a full spam/ham dataset.
    """
    if os.path.exists(DATASET_FILENAME):
        print(f"Loading data from {DATASET_FILENAME}...")
        try:
            df = pd.read_csv(DATASET_FILENAME, encoding='latin-1')
            # Assuming the dataset has columns 'v1' (label) and 'v2' (text)
            df = df[['v1', 'v2']]
            df.columns = ['Label', 'Message']
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}. Using mock data.")
            return create_mock_data()
    else:
        print(f"Dataset file '{DATASET_FILENAME}' not found. Creating mock data.")
        return create_mock_data()

def create_mock_data():
    """Creates a small DataFrame for demonstration if the real file is missing."""
    data = {
        'Label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham'],
        'Message': [
            'Go until jurong point, crazy.. Available only in bugis n great world la e buffet.',
            'WINNER!! As a valued network customer you have been selected to receive a £900 prize!',
            'Nah I dont think he goes to usf, he lives around here though',
            'Free entry in 2 a wkly competition to win FA Cup final tkts 21st May.',
            'I have been searching for the right words to thank you for this gift.',
            'Urgent! You have won a 1 year subscription to our VIP service. Call 0800...',
            "I'm gonna be in a meeting for the rest of the day",
            'Ok lor... going to watch the movie soon.'
        ]
    }
    return pd.DataFrame(data)


# --- 2. DATA PREPROCESSING (Module 1) ---

def preprocess_data(df):
    """
    Performs data cleaning, label encoding, and feature extraction (vectorization).
    """
    # 1. Label Encoding (Converts 'ham'/'spam' to 0/1)
    df['Spam_Numeric'] = df['Label'].map({'ham': 0, 'spam': 1})

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'],
        df['Spam_Numeric'],
        test_size=0.2, # Use 80% for training, 20% for testing
        random_state=42
    )
    print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 3. Feature Extraction (Text to Numerical Vectors)
    # TfidfVectorizer converts text messages into feature vectors based on word importance.
    print("Fitting TfidfVectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Features (vocabulary size): {X_train_vec.shape[1]}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# --- 3. MODEL TRAINING (Module 2) ---

def train_model(X_train_vec, y_train):
    """
    Trains the Multinomial Naive Bayes classification model.
    This is a common and efficient algorithm for text classification.
    """
    print("\n--- Training Model (Multinomial Naive Bayes) ---")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    print("Training complete.")
    return model


# --- 4. PREDICTION AND EVALUATION (Module 3) ---

def evaluate_model(model, X_test_vec, y_test):
    """
    Tests the model on the unseen test set and reports performance metrics.
    """
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_vec)
    
    # 1. Calculate Accuracy (How often the classifier is correct)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # 2. Detailed Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # 
    
    # The matrix shows:
    # Row 0 (Ham): How many true Hams were correctly classified (True Negative)
    # Row 1 (Spam): How many true Spams were correctly classified (True Positive)


# --- 5. PREDICTION EXAMPLE (Module 3 Component) ---

def make_single_prediction(model, vectorizer, text_message):
    """Predicts the label for a new, single message."""
    print(f"\n--- Prediction for New Message ---")
    print(f"Input: '{text_message}'")
    
    # Vectorize the new message using the fitted vectorizer
    new_message_vec = vectorizer.transform([text_message])
    
    # Make the prediction
    prediction = model.predict(new_message_vec)[0]
    
    # Determine confidence (probability)
    probability = model.predict_proba(new_message_vec)[0]
    
    label = 'SPAM' if prediction == 1 else 'HAM'
    confidence = max(probability) * 100
    
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.2f}%")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # 1. Load Data
    data_df = load_data()
    if data_df.empty:
        print("Cannot run without data.")
    else:
        # 2. Preprocess Data
        X_train_vec, X_test_vec, y_train, y_test, vectorizer = preprocess_data(data_df)
        
        # 3. Train Model
        nb_model = train_model(X_train_vec, y_train)
        
        # 4. Evaluate Model
        evaluate_model(nb_model, X_test_vec, y_test)
        
        # 5. Test with new examples
        make_single_prediction(nb_model, vectorizer, "Congratulations! You have been selected to win a free iPhone!")
        make_single_prediction(nb_model, vectorizer, "Hey, can we meet for lunch tomorrow?")

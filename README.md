# Spam Email/SMS Classifier (Machine Learning Mini Project)

This project is a simple machine learning model that can identify whether a given
text message is Spam or Ham (Not Spam). I created this project to learn how text
data can be processed and how machine learning models work on real-world datasets.
The whole project is written in Python and uses basic libraries like pandas,
scikit-learn, and joblib.



##  Objective of the Project
The main objective of this project is to understand:
how to preprocess text data,
how to convert text into numerical values,
how simple ML models like Naive Bayes work,
how to test and evaluate a trained model.

This project also helped me understand the basic workflow of applying machine
learning on text classification tasks.



##  Files Included in the Project
**spam_classifier.py** → Main Python code of the project  
**spam_data.csv** → Dataset file (optional, if not available a sample dataset is used)
**spam_nb_model.joblib** → Saved training model (generated after running the script)
**tfidf_vectorizer.joblib** → Saved TF-IDF vectorizer
**README.md** → Project explanation file
**STATEMENT.txt** → Project declaration for submission




##  How the Project Works

### 1. Loading the Dataset
The program tries to load a file named **spam_data.csv**.  
If the file is missing, then a small built-in sample dataset is used so that the
program can still run without errors.

### 2. Preprocessing the Data
The preprocessing steps include:
Converting the labels “ham” and “spam” into numbers (0 and 1)
Cleaning unwanted values if needed
Splitting the data into training and testing sets
Converting text messages to numerical vectors using **TF-IDF**

### 3. Training the Model
The model used in this project is the **Multinomial Naive Bayes classifier**.
This model is simple, fast, and usually performs well for text classification.

### 4. Evaluating the Model
The program prints:
Accuracy score  
Precision, Recall, F1-Score  
Confusion Matrix  

These results help us understand how well the model is working.

### 5. Predicting New Messages
The script also includes a function to test new messages.  
You can type anything and the model will predict if it is spam or ham.



##  How to Run the Project

### Step 1: Install the required libraries



### Step 2: Run the Program

### Step 3: (Optional) Add Your Own Dataset
Make a CSV file named `spam_data.csv` with the following columns:


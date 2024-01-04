    # training.py
import joblib
import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
def train_model():
    
    X_train, y_train = load_data()
    # Define and train your model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    # Save the trained model to a file
    joblib.dump(model, 'src/trained_model.joblib')

def load_data():
    # Load your dataset (replace 'your_dataset.csv' with the actual file path)
    df = pd.read_json('data\preprocessed_data.json', lines=True)

    # Assuming your dataset has a 'target' column for labels
    X = df['headline']
    y = df['category']

   
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text data
    vectorizer_nb = CountVectorizer()
    X_train_vectorized = vectorizer_nb.fit_transform(X_train)

    X_test.to_csv('data/test_data.csv', index = False)
    pd.Series(y_test).to_csv('data/test_labels.csv', index= False)

    joblib.dump(vectorizer_nb, 'src/vectorizer_nb.joblib')
    return X_train_vectorized, y_train

if __name__ == "__main__":
    train_model()
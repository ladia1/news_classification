# inference.py
import joblib
import pandas as pd  
from sklearn.metrics import classification_report
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer
def make_inference():
    # Load the trained model from the file
    model = joblib.load('src/trained_model.joblib')
    X_test = pd.read_csv('data/test_data.csv')
    
    # Vectorize the text data
    vectorizer_nb = joblib.load('src/vectorizer_nb.joblib')
    X_test_vectorized = vectorizer_nb.transform(X_test)
    # X_test = vectorizer_nb.fit_transform(X_test)
    
    
    
    y_test = pd.read_csv('data/test_labels.csv')

    y_pred = model.predict(X_test_vectorized)
    print(X_test_vectorized)
    print(y_pred)
    report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(report)

    # save the results
    pd.DataFrame({'True Labels': y_test, 'Predicted Labels': y_pred}).to_csv('predictions.csv', index=False)
if __name__ == "__main__":
    make_inference()

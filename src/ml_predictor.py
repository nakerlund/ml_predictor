# ml_predictor.py

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load

def load_data(filepath):
    """
    Load dataset from a specified filepath.
    """
    # Assuming the dataset is in CSV format
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Preprocess the data: split into features and target, and then into training and test sets.
    """
    X = data[['temperature', 'weight']]  # Features
    y = data['target']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a machine learning model on the training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    """
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

def make_prediction(model, temperature, weight):
    """
    Use the trained model to make a prediction based on input features.
    """
    input_features = np.array([[temperature, weight]])
    prediction = model.predict(input_features)
    return prediction

def save_model(model, filename='../models/trained_model.joblib'):
    """
    Save the trained model to a file.
    """
    dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='../models/trained_model.joblib'):
    """
    Load a trained model from a file.
    """
    return load(filename)

if __name__ == "__main__":
    # Example usage
    filepath = '../data/dataset.csv'  # Update this path to your dataset's location
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model)

    # Load the model (for demonstration, typically done in a different script or session)
    model_loaded = load_model()
    
    # Making a prediction with the loaded model
    temperature = 25  # example temperature
    weight = 50  # example weight
    prediction = make_prediction(model_loaded, temperature, weight)
    print(f"Predicted target value for temperature {temperature} and weight {weight}: {prediction[0]}")

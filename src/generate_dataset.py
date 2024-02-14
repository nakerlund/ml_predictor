import pandas as pd
import numpy as np
import os

def generate_mocked_data(n_samples=100):
    """
    Generate a mocked dataset with a linear relationship: target = temperature * weight.
    """
    temperature = np.random.uniform(0, 30, n_samples)
    weight = np.random.uniform(0, 100, n_samples)
    target = temperature * weights
    data = pd.DataFrame({
        'temperature': temperature,
        'weight': weight,
        'target': target
    })
    return data

def save_data_to_csv(data, filepath='../data/dataset.csv'):
    """
    Save the generated dataset to a CSV file.
    """
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    data = generate_mocked_data(n_samples=100)
    save_data_to_csv(data)

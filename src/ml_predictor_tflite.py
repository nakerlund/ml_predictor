import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from the specified filepath."""
    data = pd.read_csv(filepath)
    return data

def create_model():
    """Create and compile a TensorFlow model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[2])  # Assuming 2 input features
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def convert_to_tflite(model):
    """Convert the TensorFlow model to TFLite format with optimizations."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(tflite_model, filename='model.tflite'):
    """Save the TFLite model to a file."""
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {filename}")

def load_tflite_model(filename='model.tflite'):
    """Load a TFLite model from a file."""
    with open(filename, 'rb') as f:
        model_content = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    return interpreter

def make_prediction(interpreter, input_data):
    """Make a prediction using the loaded TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

if __name__ == "__main__":
    # Load the dataset
    data = load_data('../data/dataset.csv')
    features = data[['temperature', 'weight']].values.astype(np.float32)
    target = data['target'].values.astype(np.float32)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save the mean and std of the scaler
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    # Train the model
    model = create_model()
    model.fit(features_scaled, target, epochs=10, batch_size=1)

    # Convert to TFLite and save
    tflite_model = convert_to_tflite(model)
    save_tflite_model(tflite_model, 'model.tflite')

    # Manually apply scaling to the test input and ensure it's of type float32
    test_input_raw = np.array([[25, 50]], dtype=np.float32)  # Raw input
    test_input_scaled = (test_input_raw - scaler_mean) / scaler_scale  # Apply scaling
    test_input_scaled = test_input_scaled.astype(np.float32)  # Ensure the scaled input is float32

    # Load the TFLite model and make predictions with scaled input
    interpreter = load_tflite_model('model.tflite')
    # Ensure the input data is correctly shaped for the TFLite model
    prediction = make_prediction(interpreter, test_input_scaled)
    print(f"Predicted target value for scaled input {test_input_raw}: {prediction}")

import numpy as np
import cv2
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("model_trained.h5")


# Function to preprocess the image (same as during training)
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.equalizeHist(img)  # Apply histogram equalization
    img = img / 255.0  # Normalize pixel values
    return img


# Function to make a prediction on a new image
def predict_image(img_path, model):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))  # Resize image to 32x32 pixels
    img = preProcessing(img)  # Apply the preprocessing
    img = img.reshape(1, 32, 32, 1)  # Reshape to fit model input shape

    # Predict the class
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    return predicted_class, prediction


# Example usage
image_path = 'img004-00001.png'  # Replace with your image path
predicted_class, prediction_probabilities = predict_image(image_path, model)

print(f"Predicted class: {predicted_class}")
print(f"Prediction probabilities: {prediction_probabilities}")

from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import glob

# Define the path to the saved model
MODEL_PATH = 'cifar10_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize Flask application
app = Flask(__name__)

# Define a function to predict class for an image
def predict_class(image_path, model):
    image = load_img(image_path, target_size=(32, 32))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions)
    confidence_score = np.max(predictions)
    predicted_class = class_names[predicted_class_idx]
    return predicted_class, confidence_score

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to predict image class
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    
    # Save the image temporarily
    image_path = 'temp.jpg'
    file.save(image_path)
    
    # Perform prediction
    predicted_class, confidence_score = predict_class(image_path, model)
    
    # Remove the temporary image file
    os.remove(image_path)
    
    # Return the prediction as JSON response
    response = {
        'class': predicted_class,
        'confidence': float(confidence_score)
    }
    return jsonify(response)

# Define route to predict classes for all images in the test folder
@app.route('/predict_batch', methods=['GET'])
def predict_batch():
    # Define the path to the test folder
    test_folder = 'test_images/'
    image_paths = glob.glob(os.path.join(test_folder, '*'))
    
    predictions = []
    for image_path in image_paths:
        predicted_class, confidence_score = predict_class(image_path, model)
        predictions.append({
            'image': os.path.basename(image_path),
            'class': predicted_class,
            'confidence': float(confidence_score)
        })
    
    return jsonify(predictions)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

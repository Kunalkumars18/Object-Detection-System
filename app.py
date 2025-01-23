import os

import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Configure upload and result folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLO model
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded!", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file!", 400

    # Save uploaded file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Perform object detection
    results = model(image_path)
    annotated_image = results[0].plot()

    # Save the annotated image
    result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
    cv2.imwrite(result_path, annotated_image)

    # Return the result
    return render_template('index.html', uploaded_image=file.filename, result_image=file.filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

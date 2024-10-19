import cv2
import numpy as np
import tensorflow as tf
from Sudokuhelper import *
from flask import Flask, request, jsonify, send_file,render_template
import streamlit as st
import os
app = Flask(__name__)
# image_path='1.jpg'
# height_image=450
# width_image=450
# Test Score = 0.03514572232961655
# Test Accuracy = 0.9882772564888
model=initializemodel('model_trained_new3.h5')
# Preparing the image
# img=cv2.imread(image_path)


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
SOLVED_FOLDER = 'static/solved/'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SOLVED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save the uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # Call the Sudoku solver function here (add your own logic)
    output_image_path = os.path.join(SOLVED_FOLDER, 'solved_' + file.filename)
    SudokuSolver(image_path, output_image_path)  # Assuming a function that processes Sudoku
    
    return render_template('upload.html', input_image='uploads/' + file.filename, output_image='solved/solved_' + file.filename)

def solve_sudoku_image(input_image_path, output_image_path):
    # This function should contain your Sudoku solver logic
    # For now, it will just copy the input image to the output path
    img = cv2.imread(input_image_path)
    cv2.imwrite(output_image_path, img)

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from keras.models import load_model

model = load_model("my_model111.h5")

def save_uploaded_file(file):
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path

def preprocess_image(img):
    img_resized = cv2.resize(img, (256, 256))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    thresholded_image = np.where(img_gray > 180, 255, 0)
    cropped_image = thresholded_image[:, :]
    img_normalized = cropped_image / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)
    return img_batch


def calculate_white_area(img):
    img_resized = cv2.resize(img, (256, 256))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    cropped_image = thresholded_image[20:200, 10:250]
    contours, _ = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_white_area = sum(cv2.contourArea(contour) for contour in contours)
    return total_white_area

def categorize_image(img):
    threshold_small = 1000
    threshold_large = 2500
    total_white_area = calculate_white_area(img)
    if total_white_area < threshold_small:
        st.write("Small Amount of Cavity Detected")
        return "Demineralization"
    elif total_white_area < threshold_large:
        st.write("Medium Amount of Cavity Detected")
        return "Cavitation - requires tooth filling"
    else:
        st.write("Large Amount of Cavity Detected")
        return "Pulp Involvement - requires tooth cap"
    

@st.cache_data
def prediction(img):    
    file_path = save_uploaded_file(img)
    img = cv2.imread(file_path)
    processed_image = preprocess_image(img)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)

    if predicted_class == 0:
        return "No cavity detected"
    else:
        return categorize_image(img)
    

st.header('Cavity Detector')

img = st.file_uploader("Upload a file...", type=["jpg", "jpeg", "png"])

if img is not None:
    st.write("Uploaded File:")
    if img.type.startswith('image'):
        st.image(img, caption='Uploaded Image')
        folder = 'temp_images'
    else:
        st.error("Unsupported file format.")
        st.stop()

    if st.button("Check for Cavity"):
        result = prediction(img)
        st.write(result)
        #data = {
        #'Accuracy': [ 0.9875, 0.968 , 0.966 , 0.971],
        #'Precision': [0.783, 0.788, 0.778, 0.798],
        #'Recall': [0.374, 0.777, 0.934, 0.879],
        #}
        
        # Create a DataFrame
        #df = pd.DataFrame(data)
        #st.dataframe(df)

        graph = Image.open("1.png")
        st.image(graph, caption='Accuracy Graph', use_column_width=True)
        graph1 = Image.open("2.png")
        st.image(graph1, caption='Loss Graph', use_column_width=True)

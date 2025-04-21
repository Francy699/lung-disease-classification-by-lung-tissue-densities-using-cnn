import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image for the main app
set_background(r'C:\Users\USER\Downloads\1082_Samhitha_Backup_ung diseases classification by analysis of lung tissue densities_28.01.2025\Sourcecode\background\6.webp')

# Streamlit app title
st.title("Lung Diseases Classification By Analysis Of Lung Tissue Densities")

# Load the model
model = load_model("Model.h5")

# Define image dimensions and categories
WIDTH, HEIGHT = 65, 65
categories = ['Lung Disease', 'Normal']

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    image = np.array(image)
    
    # Ensure the image has 3 channels (RGB)
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    test_image = cv2.resize(image, (WIDTH, HEIGHT))
    test_data = np.array(test_image, dtype="float") / 255.0
    test_data = test_data.reshape([-1, WIDTH, HEIGHT, 3])
    return image, test_data

# Function to segment the image using thresholding
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(image, image, mask=thresholded)
    return segmented_image

# Streamlit interface
st.write("Upload images to classify lung diseases.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions_list = []  # Initialize an empty list to store predictions
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Load image with PIL
            image = Image.open(uploaded_file)
            
            # Display the uploaded image
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            test_image_o, test_data = load_and_preprocess_image(image)
            
            # Make prediction
            pred = model.predict(test_data)
            predictions = np.argmax(pred, axis=1)  # return to label
            
            # Append prediction to the list
            predictions_list.append(categories[predictions[0]])
            
            # Add a background color for the prediction section
            st.markdown(
                """
                <style>
                .prediction-section {
                    background-color: #f0f2f6;  /* Light gray background color */
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid #ddd;
                    margin-top: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Wrap the prediction section in a div with the custom background color
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            
            # Display the prediction
            st.subheader("Prediction")
            st.write(f'**Prediction:** {categories[predictions[0]]}')
            
            # Display the image with the prediction title
            st.subheader("Predicted Image")
            fig = plt.figure()
            fig.patch.set_facecolor('xkcd:white')
            plt.title(categories[predictions[0]])
            plt.imshow(cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
            st.pyplot(fig)
            
            # Segment the image
            segmented_image = segment_image(test_image_o)
            
            # Display the segmented image
            st.subheader("Segmented Image")
            fig = plt.figure()
            fig.patch.set_facecolor('xkcd:white')
            plt.title('Segmented Image')
            plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
            st.pyplot(fig)
            
            # Close the prediction section div
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert predictions list to string and create audio
    predictions_text = ' '.join(predictions_list)
    language = 'en'
    speech = gTTS(text=predictions_text, lang=language, slow=False)
    speech.save("sample.mp3")
    audio_path = "sample.mp3"  # Replace with the path to your MP3 audio file

    # Play audio feedback
    st.subheader("Audio Feedback")
    st.audio(audio_path, format='audio/mp3')
else:
    st.write("Please upload image files to get predictions.")
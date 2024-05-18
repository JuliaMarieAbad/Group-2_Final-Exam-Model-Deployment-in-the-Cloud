
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gdown

model_url = 'https://drive.google.com/uc?id=1-IQghpVqloccTnJJVD8sJiI3kuYsaz-8'  
model_path = 'cat_dog_classifier.h5'

@st.cache(allow_output_mutation=True)
def load_model():
    gdown.download(model_url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

class_names = ['cat', 'dog']

def predict(image):
    image = image.resize((256, 256))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = class_names[int(np.round(prediction[0]))]
    return predicted_class

st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="centered">
        <img src="dsv.jpg" width="500">
    </div>
    """,
    unsafe_allow_html=True
)
#st.image("dsv.jpg", width=500)
st.title("Cat and Dog Image Classification App")
st.markdown("### Upload an image to classify it as a cat or a dog:")
uploaded_file = st.file_uploader("", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f"Prediction: {label}")

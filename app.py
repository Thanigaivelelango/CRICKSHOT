import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your saved ResNet model
resnet_model = tf.keras.models.load_model("model1.h5")

# Define class labels
class_labels = ["Drive", "Legglance", "Pullshot", "Sweep"]

# Set Streamlit app title
st.title("Cricket Shot Classifier")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_image)
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = resnet_model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the predicted class
    st.subheader("Prediction:")
    st.write(f"The predicted class is: {predicted_class}")

import streamlit as st
import numpy as np
import keras
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model
model = tf.keras.models.load_model('mnistmodel.keras')
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

def predict_digit(image):
    # Preprocess the image to match the input format of the model
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    # Predict the digit
    prediction = model.predict(image)
    return np.argmax(prediction), max(prediction[0])

st.title("Digit Recognition App")

st.write("Draw a digit on the canvas below and click 'Predict' to see the result")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    st.image(image, caption='Drawn Image', use_column_width=True)
    if st.button('Predict'):
        digit, confidence = predict_digit(image)
        st.write(f'Predicted Digit: {digit}')
        st.write(f'Confidence: {confidence:.2f}')

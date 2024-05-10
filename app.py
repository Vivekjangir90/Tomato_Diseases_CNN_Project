import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Load the trained model
model = models.load_model('your_model.h5')

# Streamlit app
def main():
    st.title('Tomato Disease Classification')

    # File uploader for image input
    uploaded_file = st.file_uploader('Upload an image of a tomato', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.image(uploaded_file, caption='Uploaded Image', width=256)

        # Make predictions
        img_array = load_and_preprocess_image(uploaded_file, target_size=(256, 256))*255
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        # Display the predicted class
        class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']  # Define your class names here
        st.write(f'Predicted Class: {class_names[predicted_class]}')



if __name__ == '__main__':
    main()

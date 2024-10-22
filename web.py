import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('/content/trained_plant_disease_model.keras')
    test_image = test_image.resize((128, 128))  # Resize the image to your model's input size
    input_arr = tf.keras.preprocessing.image.img_to_array(test_image)  # Convert image to array
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    input_arr = input_arr / 255.0  # Normalize image
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/content/home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    """)
    
# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes.
    """)
    
# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the image file using PIL
        image = Image.open(uploaded_file)
        
        # Show the uploaded image
        st.image(image, width=400, use_column_width=True)
        
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(image)
            
            # Reading Labels
            class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                           'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew', 
                           'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                           'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy', 
                           'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                           'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                           'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                           'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                           'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                           'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                           'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                           'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                           'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                           'Tomato___healthy']
            
            st.success("Model is predicting it's a {}".format(class_names[result_index]))

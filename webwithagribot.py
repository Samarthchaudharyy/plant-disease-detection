import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import re
from nltk.stem.porter import PorterStemmer
import random
import json

# Load the crop recommendation model and scalers
randclf = pk.load(open('model.pkl', 'rb'))
mx = pk.load(open('minmaxscaler.pkl', 'rb'))
sc = pk.load(open('standscaler.pkl', 'rb'))

# Load the fertilizer recommendation model
fertilizer_model = pk.load(open('classifier1.pkl', 'rb'))

# Load Intent and Entity models
loadedIntentClassifier = load_model('saved_state/intent_model.h5')
loaded_intent_CV = pk.load(open('saved_state/IntentCountVectorizer.sav', 'rb'))
loadedEntityCV = pk.load(open('saved_state/EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = pk.load(open('saved_state/entity_model.sav', 'rb'))

# Load intents JSON
with open('/content/intents.json') as json_data:
    intents = json.load(json_data)

# Define a function for entity extraction
def getEntities(query):
    query = loadedEntityCV.transform(query).toarray()
    response_tags = loadedEntityClassifier.predict(query)
    entity_list = []
    for tag in response_tags:
        if tag in entity_label_map.values():
            entity_list.append(list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)])
    return entity_list

# Define a function for predicting user intent
def predict_intent(user_query):
    query = re.sub('[^a-zA-Z]', ' ', user_query)
    query = query.split(' ')
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]
    processed_text = ' '.join(tokenized_query)
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()
    predicted_intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_intent, axis=1)[0]
    user_intent = None
    for key, value in intent_label_map.items():
        if value == result:
            user_intent = key
            break
    return user_intent

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Crop Recommendation", "Fertilizer Recommendation", "Agriculture Chatbot"])

# Home Page
if app_mode == "Home":
    st.header("KISHAN MITRA")
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

# Disease Recognition Page
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
            class_names = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
                           'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 
                           'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                           'Corn_(maize)Common_rust', 'Corn(maize)_Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                           'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 
                           'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot',
                           'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 
                           'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
                           'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                           'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                           'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                           'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 
                           'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
                           'Tomato___healthy']
            
            st.success(f"Model is predicting it's a {class_names[result_index]}")

# Crop Recommendation Page
elif app_mode == "Crop Recommendation":
    st.header("Crop Recommendation")
    
    # Input fields for crop recommendation
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    
    if st.button("Get Crop Recommendation"):
        recommended_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        crop_dict = {
            1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya', 7: 'Orange',
            8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes', 12: 'Mango', 13: 'Banana', 
            14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans', 
            19: 'Pigeonpeas', 20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'
        }
        st.success(f"Recommended Crop: {crop_dict[recommended_crop]}")

# Fertilizer Recommendation Page
elif app_mode == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation")
    
    # Input fields for fertilizer recommendation
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    
    if st.button("Get Fertilizer Recommendation"):
        fertilizer = fertilizer_recommendation(N, P, K)
        st.success(f"Recommended Fertilizer: {fertilizer}")

# Agriculture Chatbot Page
elif app_mode == "Agriculture Chatbot":
    st.header("Agriculture Chatbot")
    user_input = st.text_input("You: ", "")
    
    if st.button("Send"):
        if user_input:
            user_intent = predict_intent(user_input)
            response = ""
            if user_intent:
                for intent in intents:
                    if 'tag' in intent and intent['tag'] == user_intent:
                        response = random.choice(intent['responses'])
                        break
            
            # Extract entities
            entities = getEntities([user_input])
            token_entity_map = dict(zip(entities, user_input.split()))
            
            st.success(f"Bot: {response}")
            if entities:
                st.write(f"Extracted Entities: {entities}")
            else:
                st.write("No entities extracted.")
        else:
            st.warning("Please enter a message.")

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Load model from Google Drive if not downloaded
@st.cache_resource
def load_model():
    model_path = "Best87K.h5"
    if not os.path.exists(model_path):
        with st.spinner("⏬ Downloading model from Google Drive..."):
            file_id = "1f6CCMxy5bIFliokxIWimliyy__3bsykk"  # Replace with your actual file ID
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# TensorFlow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0  # Normalize
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    max_confidence = np.max(predictions)
    result_index = np.argmax(predictions)
    return result_index, max_confidence

# Class names list (49 classes)
class_name = [  # Keep this as is (truncated here for brevity)
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Olive_Aculus_olearius', 'Olive_Anthracnose', 'Olive_Fusarium Wilt', 'Olive_Healthy',
    'Olive_OVYaV', 'Olive_Olive Knot', 'Olive_Olive fruit fly', 'Olive_Peacock Spots', 'Olive_Sooty Mold',
    'Olive_Verticillium Wilt', 'Olive_xylella fastidiosa', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Recommendations dictionary (you can expand this further)
recommendations = {
    'Apple___Apple_scab': "Use fungicides containing captan or myclobutanil. Prune infected leaves.",
    'Apple___Black_rot': "Remove and destroy mummified fruits. Apply fungicide sprays during bloom.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper hosts. Apply protective fungicides.",
    'Apple___healthy': "No action needed. Keep monitoring regularly.",
    'Blueberry___healthy': "Plant looks healthy. Maintain soil pH and irrigation.",
    # ... include all other mappings like in your original code ...
    'Tomato___healthy': "Healthy tomato plant. Maintain balanced fertilization."
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.image("Header.png", width=200)
    st.header("AgriVision AI V1.0 - PROTOTYPE")
    st.image("home.jpg", use_container_width=True)
    st.markdown("""
    Welcome to AgriVision AI V1.0, a Plant Disease Recognition System! 🌿🔍

    Designed by **NAWA's AI-Division**, this tool helps identify plant diseases efficiently.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page.
    2. **Analysis:** Our AI model analyzes the image.
    3. **Results:** View results and recommendations.

    ### Why Choose Us?
    - **Accurate:** Trained on over 74,000 images across 49 classes.
    - **User-Friendly:** Clean, responsive interface.
    - **Fast & Efficient:** Predictions in seconds.

    Head over to the **Disease Recognition** tab to try it out!
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    ### 🧠 Model Overview
    This model is trained on a dataset of **74,016 images** in **49 classes** using deep learning.

    ### 🌱 Supported Crops
    - 🍎 Apple, 🫐 Blueberry, 🍒 Cherry, 🌽 Corn, 🍇 Grape, 🫒 Olive
    - 🍊 Orange, 🍑 Peach, 🫑 Bell Pepper, 🥔 Potato, 🍓 Strawberry
    - 🫘 Soybean, 🧅 Squash, 🍇 Raspberry, 🍅 Tomato

    ### 📁 Dataset Summary
    - Training Images: 74,016
    - Classes: 49
    - Image Size: 128x128 RGB
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image of a plant leaf (JPG or PNG):", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("🔎 Analyzing image..."):
                result_index, confidence = model_prediction(test_image)

            st.write(f"📊 Prediction Confidence: **{confidence:.2f}**")

            if confidence < 0.5:
                st.warning("⚠️ Low confidence. Try a clearer image or a different angle.")
            elif 0 <= result_index < len(class_name):
                disease = class_name[result_index]
                st.success(f"✅ Prediction: **{disease}**")
                recommendation_message = recommendations.get(disease, "🧪 No recommendation available.")
                st.info(f"💡 Recommendation: {recommendation_message}")
            else:
                st.error("❌ Invalid prediction index. Model/data mismatch.")

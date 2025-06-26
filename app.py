import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# --- Load model from Google Drive ---
@st.cache_resource
def load_model():
    model_path = "Best87K.h5"
    if not os.path.exists(model_path):
        with st.spinner("⏬ Downloading model from Google Drive..."):
            file_id = "1f6CCMxy5bIFliokxIWimliyy__3bsykk"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False, use_cookies=True)
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Predict function ---
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)


# Class names list (49 classes)
class_name = [
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

# Recommendations dictionary
recommendations = {
    'Apple___Apple_scab': "Use fungicides containing captan or myclobutanil. Prune infected leaves.",
    'Apple___Black_rot': "Remove and destroy mummified fruits. Apply fungicide sprays during bloom.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper hosts. Apply protective fungicides.",
    'Apple___healthy': "No action needed. Keep monitoring regularly.",
    'Blueberry___healthy': "Plant looks healthy. Maintain soil pH and irrigation.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur-based fungicides. Remove infected leaves.",
    'Cherry_(including_sour)___healthy': "No disease detected. Keep monitoring.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops and apply fungicides.",
    'Corn_(maize)___Common_rust_': "Plant resistant hybrids. Apply fungicides if infection is severe.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids and manage crop residues.",
    'Corn_(maize)___healthy': "Healthy plant. Continue regular management.",
    'Grape___Black_rot': "Prune affected vines. Apply protective fungicides early in the season.",
    'Grape___Esca_(Black_Measles)': "Remove and destroy affected wood. Maintain vine health.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply copper fungicides and maintain canopy airflow.",
    'Grape___healthy': "No signs of disease. Maintain good vineyard practices.",
    'Olive_Aculus_olearius': "Use acaricides and monitor closely.",
    'Olive_Anthracnose': "Prune infected branches. Apply fungicides during wet seasons.",
    'Olive_Fusarium Wilt': "Remove infected plants and improve soil drainage.",
    'Olive_Healthy': "Healthy olive tree. Maintain good irrigation.",
    'Olive_OVYaV': "Control vector insects and remove infected material.",
    'Olive_Olive Knot': "Prune affected branches. Avoid injury during pruning.",
    'Olive_Olive fruit fly': "Use traps and insecticides targeted to fruit fly.",
    'Olive_Peacock Spots': "Apply copper fungicides in autumn and winter.",
    'Olive_Sooty Mold': "Control scale insects and clean leaves.",
    'Olive_Verticillium Wilt': "Remove infected trees. Avoid replanting susceptible cultivars.",
    'Olive_xylella fastidiosa': "Follow quarantine rules and control insect vectors.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees. Control psyllid vectors.",
    'Peach___Bacterial_spot': "Apply copper-based bactericides. Remove infected twigs.",
    'Peach___healthy': "Healthy peach tree. Maintain nutrient balance.",
    'Pepper,_bell___Bacterial_spot': "Use certified seeds and rotate crops.",
    'Pepper,_bell___healthy': "Plant looks healthy. Monitor for pests.",
    'Potato___Early_blight': "Apply fungicides preventatively. Remove infected foliage.",
    'Potato___Late_blight': "Use certified disease-free seed. Apply fungicides like mancozeb.",
    'Potato___healthy': "Healthy plant. Maintain good watering practices.",
    'Raspberry___healthy': "No disease detected. Maintain good field hygiene.",
    'Soybean___healthy': "Healthy crop. Rotate crops to reduce disease risk.",
    'Squash___Powdery_mildew': "Apply sulfur or potassium bicarbonate sprays.",
    'Strawberry___Leaf_scorch': "Remove infected leaves and apply fungicides.",
    'Strawberry___healthy': "Healthy strawberry plant. Maintain soil moisture.",
    'Tomato___Bacterial_spot': "Avoid overhead irrigation. Use copper-based sprays.",
    'Tomato___Early_blight': "Apply fungicides early and remove infected leaves.",
    'Tomato___Late_blight': "Use resistant varieties and fungicide sprays.",
    'Tomato___Leaf_Mold': "Improve ventilation and apply fungicides.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves and apply fungicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides and encourage natural predators.",
    'Tomato___Target_Spot': "Apply fungicides and remove crop debris.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whitefly vectors and remove infected plants.",
    'Tomato___Tomato_mosaic_virus': "Use virus-free seeds and resistant varieties.",
    'Tomato___healthy': "Healthy tomato plant. Maintain balanced fertilization."
}

# --- Sidebar ---
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# --- Home Page ---
if app_mode == "Home":
    st.image("Header.png", width=200)
    st.header("AgriVision AI V1.0 - PROTOTYPE")
    st.image("home.jpg", use_container_width=True)
    st.markdown("""
    Welcome to AgriVision AI 🌿🔍  
    Upload a plant leaf image to identify potential diseases and get recommendations.
    """)

# --- About Page ---
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    Deep learning model trained on 74,000+ leaf images across 49 classes.  
    Helps farmers detect and treat diseases in crops quickly.
    """)

# --- Disease Recognition ---
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload a plant leaf image:")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            with st.spinner("Analyzing image..."):
                
                result_index, confidence = model_prediction(test_image)

            st.write(f"📊 Confidence: **{confidence:.2f}**")

            if confidence < 0.5:
                st.warning("⚠️ The model does NOT confidently recognize this disease.")
                st.info("Please try with a clearer or different leaf image.")
            elif 0 <= result_index < len(class_name):
                disease = class_name[result_index]
                st.success(f"✅ Prediction: **{disease}**")
                recommendation_message = recommendations.get(disease, "🧪 No specific recommendation available yet.")
                st.info(f"💡 Recommendation: {recommendation_message}")
            else:
                st.error("❌ Prediction index is invalid.")

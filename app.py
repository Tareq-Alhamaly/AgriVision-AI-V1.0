import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="Best87K_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Run inference using TFLite
def model_prediction_tflite(test_image, interpreter):
    image = Image.open(test_image).resize((128, 128)).convert("RGB")
    input_arr = np.array(image, dtype=np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    result_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return result_index, confidence

# Class names list
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

# Recommendations dictionary (same as earlier — omitted here for brevity)
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

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.image("Header.png", width=200)

    st.header("AgriVision AI V1.0 - PROTOTYPE-")
    image_path = "home.jpg"
    st.image(image_path, use_container_width=True)

    st.markdown("""
    Welcome to AgriVision AI V1.0, a Plant Disease Recognition System! 🌿🔍

    Designed, tested & deployed by **NAWA's** AI-Division to help identify plant diseases efficiently.
    Upload an image of a plant, and our system will analyze it to detect any signs of disease.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page.
    2. **Analysis:** Our AI model analyzes the image.
    3. **Results:** View results and recommendations.

    ### Why Choose Us?
    - **Accurate:** Trained on over 74,000 images across 49 classes.
    - **User-Friendly:** Clean, responsive interface.
    - **Fast & Efficient:** Predictions in seconds.
    ---

    ### 💼 Contact Information

    - 📧 **Emails**:
        - [tarek.alhamaly@nawa-eng.com.ly](mailto:tarek.alhamaly@nawa-eng.com.ly)
        - [firas.m@nawa-eng.com.ly](mailto:firas.m@nawa-eng.com.ly)



    - 📞 **Phone Numbers**:
        - +218 91 788 0952
        - +216 28 163 411

    ---



    ### 📄 Download Project Summary (PDF)

    You can download the project brochure or documentation here:

    """)

    # PDF download button
    with open("Datasheet.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="📥 Download PDF",
                       data=PDFbyte,
                       file_name="AgriVision_Project_Summary.pdf",
                       mime='application/pdf')

    st.markdown("""
    ---

    ### ✅ Why Choose AgriVision AI?

    - ✅ **High Accuracy**: Trained on over **74,000 images** across **49 plant disease and health classes**.
    - 💻 **User-Friendly**: Clean, responsive web interface powered by **Streamlit**.
    - ⚡ **Fast Inference**: Get predictions in seconds — even on low-resource devices.

    ---

    ### 📍 Get Started

    👉 Click on the **Disease Recognition** tab in the sidebar to begin!
    """)


# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    ### 🧠 Model Overview
    This model is trained on a diverse dataset of **74,016 images** belonging to **49 classes**.
    It uses deep learning to detect various **plant diseases** and distinguish them from healthy samples.

    ### 🌱 Supported Plant Types

    The system supports a wide range of crops and fruits, including:

    #### 🍎 Apple
    - Apple scab, Black rot, Cedar apple rust, Healthy

    #### 🫐 Blueberry
    - Healthy

    #### 🍒 Cherry (incl. sour)
    - Powdery mildew, Healthy

    #### 🌽 Corn (maize)
    - Cercospora leaf spot, Common rust, Northern leaf blight, Healthy

    #### 🍇 Grape
    - Black rot, Esca (Black Measles), Leaf blight, Healthy

    #### 🫒 Olive
    - Aculus olearius, Anthracnose, Fusarium Wilt, Peacock Spots, Verticillium Wilt, Xylella fastidiosa, Olive Knot,
      Olive fruit fly, Sooty Mold, OVYaV virus, Healthy

    #### 🍊 Orange
    - Huanglongbing (Citrus Greening)

    #### 🍑 Peach
    - Bacterial spot, Healthy

    #### 🫑 Bell Pepper
    - Bacterial spot, Healthy

    #### 🥔 Potato
    - Early blight, Late blight, Healthy

    #### 🍓 Strawberry
    - Leaf scorch, Healthy

    #### 🫘 Soybean, 🧅 Squash, 🍇 Raspberry
    - Healthy

    #### 🍅 Tomato
    - Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot,
      Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

    These classes were chosen based on real agricultural threats across various regions.

    ### 📁 Dataset Summary
    - **Training images:** 74,016
    - **Classes:** 49
    - **Input size:** 128x128 RGB
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image of a plant leaf:")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔎 Analyzing image...")

            interpreter = load_tflite_model()
            result_index, confidence = model_prediction_tflite(test_image, interpreter)

            st.write(f"📊 Prediction Confidence: **{confidence:.2f}**")

            if confidence < 0.5:
                st.warning("⚠️ The model does NOT confidently recognize this disease.")
                st.info("Try with a clearer or different image.")
            elif 0 <= result_index < len(class_name):
                disease = class_name[result_index]
                st.success(f"✅ Model predicts: **{disease}**")

                recommendation_message = recommendations.get(disease, "🧪 No specific recommendation available.")
                st.info(f"💡 Recommendation: {recommendation_message}")
            else:
                st.error("❌ Invalid prediction index.")

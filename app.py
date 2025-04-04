import streamlit as st
import google.generativeai as genai
from PIL import Image

# Configure Gemini with Streamlit secret
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Gemini Vision model
model = genai.GenerativeModel("gemini-pro-vision")

st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")
st.title("🌱 Plant Disease Detection using Gemini Vision")
st.markdown("Upload a leaf image to detect possible plant diseases using Google's Gemini Vision AI.")

# Upload the image
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prompt = "This is an image of a plant leaf. Please identify if there's any plant disease and suggest treatment. If it's healthy, say that too."

    try:
        st.info("🔍 Analyzing image with Gemini...")
        response = model.generate_content([prompt, image])
        st.success("✅ Gemini Result:")
        st.write(response.text)
    except Exception as e:
        st.error(f"❌ Error: {e}")

import streamlit as st
import google.generativeai as genai
from PIL import Image
from google.generativeai.types import content_types

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Set up the model
model = genai.GenerativeModel("gemini-pro-vision")

st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")
st.title("🌱 Plant Disease Detection using Gemini Vision")
st.markdown("Upload a leaf image to detect possible plant diseases using Google's Gemini Vision AI.")

# Upload image
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Read image as bytes
    image_bytes = uploaded_file.read()

    # Convert to Gemini-supported image part
    image_part = content_types.ImagePart.from_bytes(image_bytes, mime_type="image/jpeg")

    prompt = "This is an image of a plant leaf. Please identify if there's any plant disease and suggest a cure. If the plant is healthy, mention that too."

    try:
        st.info("🔍 Analyzing with Gemini Vision...")
        response = model.generate_content([prompt, image_part])
        st.success("✅ Gemini Result:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error: {e}")

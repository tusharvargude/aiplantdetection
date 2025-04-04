import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64

# Load Gemini API key from secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Setup model
model = genai.GenerativeModel('gemini-pro-vision')

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection using Google Gemini")
st.markdown("Upload a leaf image to detect possible plant diseases.")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Read image content
    image_bytes = uploaded_file.read()

    # Ask Gemini to analyze it
    st.info("🧠 Analyzing image with Gemini...")
    response = model.generate_content([
        "This is an image of a plant leaf. Tell me what plant disease (if any) it has, how to cure it, and if it's healthy, say so.",
        image_bytes
    ])

    st.success("✅ Analysis Result")
    st.write(response.text)

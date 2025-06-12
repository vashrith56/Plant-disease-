from PIL import Image
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the GenAI API
api_key = "AIzaSyBghYvHAna4dG_M9iwKA6Euz6-GVPvl02I"
if not api_key:
    st.error("Missing API key. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Function to get response from GenAI model
def get_gemini_response(input_prompt, image_data, mime_type):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    # Prepare the input as a Content object with parts
    content_input = {
        "parts": [
            {"text": input_prompt},  # Text part for the prompt
            {
                "mime_type": mime_type,  # MIME type of the image
                "data": image_data  # Raw image byte data
            }
        ]
    }
    
    # Generate content using the prepared input
    response = model.generate_content(content_input)
    return response.text

# Streamlit app setup
st.set_page_config(page_title="Calorie Advisor App")
st.header("Calorie Advisor App")

# Upload image
upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    try:
        # Read image byte data and MIME type
        image_data = upload_file.read()
        mime_type = upload_file.type  # e.g., "image/jpeg" or "image/png"
        
        # Display the image
        image = Image.open(upload_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Define prompt
input_prompt = """
You are an expert nutritionist. Analyze the food items in the image and calculate the total calories. 
Provide a breakdown of every food item with its calorie intake in the following format:

1. Item 1 - X calories
2. Item 2 - Y calories
...

Also, mention whether the food is healthy or unhealthy. Provide a percentage breakdown of 
carbohydrates, fibers, sugars, proteins, fats, and other essential nutrients in the diet.
"""

# Button to trigger calorie analysis
if st.button("Tell me about the total calories in the image") and upload_file is not None:
    try:
        # Get AI response
        response = get_gemini_response(input_prompt, image_data, mime_type)
        
        # Display the AI's response
        st.header("Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os

# --- 1. SETTING UP THE STYLES ---
hide_style = """
    <style>
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)

# --- 2. MODEL PREPARATION ---
class InsectModel(nn.Module):
    def __init__(self, num_classes=40):
        super(InsectModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InsectModel(num_classes=40)
    # Ensure this filename matches your folder exactly
    model.load_state_dict(torch.load("pest_model.pth", map_location=device))
    model.eval()
    return model, device

def model_prediction(test_image):
    model, device = load_model_weights()
    image = Image.open(test_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_arr = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(input_arr)
    return torch.argmax(predictions, dim=1).item()

from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)
app_mode = st.selectbox("Select a Page", ["HOME", "PEST DETECTION"])

# --- 4. HOME PAGE ---
if app_mode == "HOME":
    # 1. Main Title with green color
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AgroAI: Smart Pest Detection</h1>", unsafe_allow_html=True)
    
    # 2. Sub-headings
    st.markdown("<p style='text-align: center;'>Empowering Farmers with AI-Powered Pest Recognition.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload pest images to identify species accurately and access actionable insights.</p>", unsafe_allow_html=True)

    st.write("---")

    # 3. Features Section (3 Columns) with Safety Checks
    st.markdown("## Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists("feat-1.png"):
            st.image("feat-1.png", use_container_width=True)
        else:
            st.markdown("### 📷") # Fallback icon if file is missing
        st.markdown("<p style='text-align: center;'><b>Pest Identification</b></p>", unsafe_allow_html=True)
        st.write("Identify crop pests with AI.")

    with col2:
        if os.path.exists("feat-2.jpg"):
            st.image("feat-2.jpg", use_container_width=True)
        else:
            st.markdown("### 💡") # Fallback icon if file is missing
        st.markdown("<p style='text-align: center;'><b>Actionable Insights</b></p>", unsafe_allow_html=True)
        st.write("Get pest details and remedies.")

    with col3:
        if os.path.exists("feat-3.png"):
            st.image("feat-3.png", use_container_width=True)
        else:
            st.markdown("### ⚡") # Fallback icon if file is missing
        st.markdown("<p style='text-align: center;'><b>Real-Time Results</b></p>", unsafe_allow_html=True)
        st.write("Receive instant predictions.")

    st.write("---")

    # 4. How It Works Section
    st.markdown("## How It Works")
    st.markdown("""
    1.  Navigate to the Pest Recognition page.
    2.  Upload an image of the pest.
    3.  Get instant results along with information and treatment suggestions.
    """)

elif(app_mode=="PEST DETECTION"):
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Pest Detection</h1>", unsafe_allow_html=True)
    
    # helper function to prevent crashes if images are missing
    def display_asset(path_data, caption="", width=None):
        if not path_data: return
        path = path_data[0] if isinstance(path_data, list) else path_data
        if os.path.exists(path):
            try:
                st.image(path, caption=caption, width=width, use_container_width=False if width else True)
            except:
                pass 

    test_image = st.file_uploader("Choose an Image:", key="agro_pest_v11")
    
    if(test_image is not None):
        if(st.button("Show Image")):
            st.image(test_image, use_container_width=True)
            
        if(st.button("Predict")):
            result_index = model_prediction(test_image)
            
            json_path = 'pest.pest_details.json'
            
            # THE TRY BLOCK STARTS HERE
            try:
                if not os.path.exists(json_path):
                    st.error(f"Error: {json_path} missing.")
                else:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        details = json.load(f)
                        pest_info = details[result_index]
                        
                        st.divider()
                        st.markdown(f"<h1 style='text-align: center; color: #2e7d32;'>Detected: {pest_info.get('name')}</h1>", unsafe_allow_html=True)
                        
                        # --- THE TABS WITH STRICT IMAGE LOOKUP ---
                        tab1, tab2, tab3 = st.tabs(["🔍 About the Pest", "💊 Chemical Treatment", "🛡️ Prevention"])

                        with tab1:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.subheader("Description")
                                st.write(pest_info.get('description', 'N/A').replace('•', '\n\n•'))
                            with col2:
                                st.write("**Reference Photo**")
                                # FORCE looking in 'pest' folder for actual bug picture
                                manual_bug_path = f"pest/{result_index + 1}.jpg"
                                if os.path.exists(manual_bug_path):
                                    display_asset(manual_bug_path, width=200)
                                else:
                                    st.caption("*(Pest photo not found)*") 
                        with tab2:
                            st.subheader("Recommended Control Products")
                            
                            # 1. Show the Treatment Text
                            p_text = pest_info.get('pesticides') or pest_info.get('pesticide') or ""
                            st.info(p_text.replace('•', '\n\n•'))

                            # 2. AUTOMATIC IMAGE MATCHING (No hardcoded list)
                            target_bottle = None
                            
                            # First, check if the JSON path is already a valid pesticide bottle
                            p_path = pest_info.get('p_image')
                            if isinstance(p_path, list): p_path = p_path[0]
                            
                            if p_path and "pesticide" in str(p_path).lower() and os.path.exists(p_path):
                                target_bottle = p_path
                            
                            # If the JSON is wrong (pointing to a bug), look at your actual folder
                            elif os.path.exists("pesticides"):
                                # Get all the image names you actually have in your folder
                                available_pesticides = os.listdir("pesticides")
                                
                                for filename in available_pesticides:
                                    # Get the name without '.jpg' (e.g., 'deltamethrin')
                                    chem_name = filename.split('.')[0].lower()
                                    
                                    # If that name is mentioned in the treatment text, use that image!
                                    if chem_name in p_text.lower():
                                        target_bottle = os.path.join("pesticides", filename)
                                        break

                            # 3. DISPLAY
                            if target_bottle:
                                st.image(target_bottle, caption="Recommended Control Product", width=250)
                            else:
                                st.caption("ℹ️ No matching product bottle found in the pesticides folder.")
                        with tab3:
                            st.subheader("Preventive Actions")
                            st.success(pest_info.get('prevention', 'N/A').replace('•', '\n\n•'))

                        st.markdown("---")
                        st.caption("🌿 **Please use this as a helpful guide, but check with a local expert to be sure before starting any treatment.**")

            # THE EXCEPT BLOCK - THIS FIXES YOUR SYNTAX ERROR
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
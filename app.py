# streamlit_app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import the Generator class from your training script
# This assumes train_gan.py is in the same directory
from train_gan import Generator

# --- CONFIGURATION ---
MODEL_PATH = "saved_models/gen_epoch_499.pth" # 👈 IMPORTANT: Set this to your best model's path
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL LOADING ---
@st.cache_resource # This caches the model so it doesn't reload on every interaction
def load_model(model_path):
    """Loads the trained generator model."""
    model = Generator(in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode
    return model

# --- IMAGE PROCESSING ---
def preprocess_image(image):
    """Prepares the image for the model."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

def postprocess_image(tensor):
    """Converts the model's output tensor back to a displayable image."""
    # De-normalize from [-1, 1] to [0, 1]
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor * 0.5 + 0.5
    # Convert to PIL Image
    image = transforms.ToPILImage()(tensor)
    return image

# --- STREAMLIT UI ---

# Main page
st.title("🖼️ Image Restoration with GANs")
st.write("Upload a corrupted image and see the model work its magic!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load the model
model = load_model(MODEL_PATH)

if uploaded_file is not None:
    # Open the uploaded image
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(original_image, use_container_width=True)

    # Add a restore button
    if st.button("Restore Image", use_container_width=True):
        with st.spinner('Applying the magic... ✨'):
            # Preprocess the image and get the model's prediction
            image_tensor = preprocess_image(original_image).to(DEVICE)
            
            with torch.no_grad():
                restored_tensor = model(image_tensor)
            
            # Post-process the output to get the final image
            restored_image = postprocess_image(restored_tensor)
            
            with col2:
                st.header("Restored Image")
                st.image(restored_image, use_container_width=True)
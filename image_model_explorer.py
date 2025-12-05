import streamlit as st
import requests
from PIL import Image
import io
import json
from datasets import load_dataset
import random

# Page configuration
st.set_page_config(
    page_title="AI Image Recognition Explorer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 AI Image Recognition Explorer")
st.markdown("""
Explore how AI models classify images. Select images from different datasets and see what the model predicts.
Pay attention to where the model succeeds and where it fails!
""")

# Hugging Face API setup
API_URL = "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224"
headers = {"Authorization": f"Bearer {st.secrets.get('HUGGINGFACE_TOKEN', '')}"}

def query_model(image_bytes):
    """Send image to Hugging Face API and get predictions"""
    try:
        response = requests.post(API_URL, headers={"Content-Type": "image/jpeg", **headers}, data=image_bytes)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def load_sample_images(dataset_name, split, num_samples=20):
    """Load sample images from Hugging Face datasets"""
    try:
        if dataset_name == "cifar10":
            dataset = load_dataset("cifar10", split=split)
            images = random.sample(list(dataset), min(num_samples, len(dataset)))
            return [(img['img'], f"True label: {img['label']}") for img in images]
        
        elif dataset_name == "food101":
            dataset = load_dataset("food101", split=split, streaming=True)
            images = random.sample(list(dataset), min(num_samples, len(dataset)))
            return [(img['image'], f"True label: {img['label']}") for img in images]
        
        elif dataset_name == "imagenet-1k":
            # Using a subset for faster loading
            dataset = load_dataset("imagenet-1k", split=split, streaming=True)
            images = []
            for i, img in enumerate(dataset):
                if i >= num_samples:
                    break
                images.append((img['image'], f"True label: {img['label']}"))
            return images
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset:",
    ["cifar10", "food101", "Upload your own image"],
    help="Different datasets to explore model performance"
)

# Initialize session state for storing results
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Image Selection")
    
    selected_image = None
    image_bytes = None
    true_label = None
    
    if dataset_choice == "Upload your own image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to test the model"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_bytes = uploaded_file.getvalue()
            true_label = "Unknown (your image)"
    
    else:
        # Load dataset samples
        with st.spinner(f"Loading {dataset_choice} samples..."):
            samples = load_sample_images(dataset_choice, "train", num_samples=20)
        
        if samples:
            sample_idx = st.slider(
                "Select image number:",
                0, len(samples) - 1, 0,
                help="Slide to browse through different images"
            )
            
            selected_image, true_label = samples[sample_idx]
            
            # Convert PIL image to bytes for API
            img_byte_arr = io.BytesIO()
            selected_image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
    
    # Display selected image
    if selected_image:
        st.image(selected_image, caption="Selected Image", use_container_width=True)
        if true_label:
            st.info(f"**{true_label}**")
        
        # Classify button
        if st.button("🤖 Classify Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predictions = query_model(image_bytes)
                
                if "error" in predictions:
                    st.error(f"Error: {predictions['error']}")
                    st.info("💡 Note: The model may need a few seconds to load. Try again in a moment.")
                else:
                    st.session_state.current_predictions = predictions
                    st.session_state.current_image = selected_image
                    st.session_state.current_true_label = true_label
                    
                    # Add to history
                    st.session_state.results_history.append({
                        'true_label': true_label,
                        'predictions': predictions[:3]  # Top 3
                    })

with col2:
    st.subheader("🎯 Model Predictions")
    
    if 'current_predictions' in st.session_state:
        predictions = st.session_state.current_predictions
        
        st.markdown("### Top 5 Predictions:")
        
        for i, pred in enumerate(predictions[:5], 1):
            confidence = pred['score'] * 100
            label = pred['label']
            
            # Color code based on confidence
            if confidence > 50:
                color = "🟢"
            elif confidence > 20:
                color = "🟡"
            else:
                color = "🔴"
            
            st.markdown(f"{color} **{i}. {label}**")
            st.progress(pred['score'])
            st.caption(f"Confidence: {confidence:.1f}%")
            st.markdown("---")
        
        # Analysis questions
        st.markdown("### 🤔 Discussion Questions:")
        st.markdown("""
        - Is the top prediction correct?
        - How confident is the model?
        - Do any wrong predictions make sense?
        - What might the model be "seeing" in the image?
        - Are there biases in the predictions?
        """)
    else:
        st.info("👆 Select an image and click 'Classify Image' to see predictions")

# Results history
if st.session_state.results_history:
    with st.expander("📊 View Classification History"):
        for i, result in enumerate(reversed(st.session_state.results_history[-10:]), 1):
            st.markdown(f"**Test {len(st.session_state.results_history) - i + 1}**")
            st.write(f"True label: {result['true_label']}")
            st.write(f"Top prediction: {result['predictions'][0]['label']} "
                    f"({result['predictions'][0]['score']*100:.1f}%)")
            st.markdown("---")

# Information footer
with st.expander("ℹ️ About this Application"):
    st.markdown("""
    ### How it works:
    This application uses a Vision Transformer (ViT) model trained on ImageNet to classify images.
    
    ### Learning objectives:
    - Understand how AI models make predictions
    - Identify where models succeed and fail
    - Discuss potential biases in AI systems
    - Explore the limitations of current AI technology
    
    ### The model:
    - **Name**: google/vit-base-patch16-224
    - **Training data**: ImageNet-1k (1000 categories)
    - **Architecture**: Vision Transformer
    
    ### For discussion:
    - Why might the model fail on certain images?
    - What types of images are underrepresented in training data?
    - How might these failures affect real-world applications?
    """)

# Clear history button
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.results_history = []
    st.rerun()
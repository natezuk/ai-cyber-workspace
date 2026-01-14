import streamlit as st
#import requests
from PIL import Image
import io
import time
import os
from huggingface_hub import InferenceClient

# Configure page
st.set_page_config(page_title="AI Image Bias Explorer", layout="wide")

# Get API token from environment
##HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Title and introduction
st.title("🎨 AI Image Bias Explorer")
st.markdown("""
This workshop explores how AI image models represent the world. Generate images using the same prompts 
across different models and observe patterns in representation.
""")

# Check if token is loaded
# if not HF_TOKEN:
#     st.error("⚠️ Hugging Face API token not found!")
#     st.info("""
#     **Setup Instructions:**
#     1. Create a `.env` file in the same directory as this script
#     2. Add your token: `HUGGINGFACE_TOKEN=your_token_here`
#     3. Get a free token at: https://huggingface.co/settings/tokens
#     4. Restart the application
#     """)
#     st.stop()


client = InferenceClient(
    provider="hf-inference",
    api_key=st.secrets.get('HUGGINGFACE_TOKEN', '')
    #api_key=os.environ["HUGGINGFACE_TOKEN"],
)

# Model loading headers
#headers = {"Authorization": f"Bearer {st.secrets.get('HUGGINGFACE_TOKEN', '')}"}

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.success("✓ API token loaded from .env file")

# Model selection
MODELS = {
    #"Stable Diffusion 3.5": "stabilityai/stable-diffusion-3.5-medium",
    #"SDXL-Turbo": "stabilityai/sdxl-turbo",
    "Flux-Schnell": "black-forest-labs/FLUX.1-dev",
    "Stable Diffusion 1.0": "stabilityai/stable-diffusion-xl-base-1.0"
}

selected_models = st.sidebar.multiselect(
    "Select Models",
    list(MODELS.keys()),
    default=list(MODELS.keys())[:2]
)

# Suggested prompts organized by bias category
st.sidebar.header("📋 Suggested Prompts")
st.sidebar.markdown("Click to copy:")

PROMPT_CATEGORIES = {
    "👔 Occupational Stereotypes": [
        "a doctor",
        "a nurse", 
        "a CEO",
        "an engineer",
        "a teacher",
        "a scientist"
    ],
    "👥 Default Representations": [
        "a person",
        "a family",
        "a professional",
        "a student"
    ],
    "🌍 Cultural Assumptions": [
        "a wedding",
        "a house",
        "a traditional meal",
        "a neighborhood"
    ],
    "🎨 Creative Roles": [
        "an artist",
        "a musician",
        "a dancer",
        "a fashion designer"
    ],
    "💪 Physical Attributes": [
        "a strong person",
        "an attractive person",
        "an athlete",
        "a model"
    ]
}

for category, prompts in PROMPT_CATEGORIES.items():
    with st.sidebar.expander(category):
        for prompt in prompts:
            st.code(prompt, language=None)

# Main interface
st.header("🖼️ Generate and Compare")

# Prompt input
user_prompt = st.text_input(
    "Enter your prompt:",
    placeholder="e.g., 'a doctor' or 'a CEO'",
    help="Try simple, common occupations or roles to see how models represent them"
)

# Generation button
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    generate_button = st.button("🎨 Generate Images", type="primary", width="content")
with col2:
    num_images = st.selectbox("Images per model:", [1, 2, 3], index=0)

# Function to query Hugging Face API
def query_model(model_id, prompt):
    """Query a Hugging Face model and return the image"""
    #API_URL = f"https://router.huggingface.co/models/{model_id}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }
    }
    
    try:
        # using huggingface_hub
        image = client.text_to_image(
            prompt,
            model=model_id,
        )

        return image, None
        # response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        # if response.status_code == 503:
        #     return None, "Model is loading. Please wait and try again in a moment."
        # elif response.status_code == 200:
        #     image = Image.open(io.BytesIO(response.content))
        #     return image, None
        # else:
        #     return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

# Generate images
if generate_button:
    if not user_prompt:
        st.warning("Please enter a prompt to generate images.")
    elif not selected_models:
        st.warning("Please select at least one model.")
    else:
        st.markdown("---")
        st.subheader(f"Results for: \"{user_prompt}\"")
        
        # Create columns for each model
        model_cols = st.columns(len(selected_models))
        
        for idx, model_name in enumerate(selected_models):
            with model_cols[idx]:
                st.markdown(f"**{model_name}**")
                model_id = MODELS[model_name]
                
                with st.spinner(f"Generating..."):
                    for img_num in range(num_images):
                        image, error = query_model(model_id, user_prompt)
                        
                        if image:
                            st.image(image, width="content")
                        else:
                            st.error(error)
                        
                        # Small delay between requests to avoid rate limiting
                        if img_num < num_images - 1:
                            time.sleep(1)
        
        # Reflection questions
        st.markdown("---")
        st.subheader("🤔 Reflection Questions")
        
        with st.expander("Click to explore discussion questions"):
            st.markdown("""
            **Observe the images you generated:**
            
            1. **Representation**: What do you notice about the gender, race, age, and appearance of people in these images?
            
            2. **Consistency**: Are the representations similar across different models? What differences do you see?
            
            3. **Stereotypes**: Do the images reinforce any stereotypes? What assumptions seem to be "baked in"?
            
            4. **Missing perspectives**: Who or what is NOT represented in these images?
            
            5. **Real-world impact**: How might these biases affect real applications of AI (hiring tools, education, healthcare)?
            
            6. **Data origins**: Why do you think the models produce these particular representations?
            """)
        
# Footer with educational context
st.markdown("---")
with st.expander("ℹ️ About This Workshop"):
    st.markdown("""
    ### Understanding AI Image Bias
    
    **Why do biases exist in these models?**
    - These models are trained on billions of images from the internet
    - Internet content reflects historical and ongoing societal biases
    - Common associations in training data become "defaults" in the model
    
    **What can we learn?**
    - AI systems reflect the data they're trained on
    - "Objective" or "neutral" AI is a misconception
    - Careful evaluation and diverse perspectives are essential
    - Technical solutions alone cannot solve social problems
    
    **Important notes:**
    - Different models may show different biases based on their training data
    - Results can vary between generations due to randomness
    - These biases have real-world consequences when AI is deployed
    """)

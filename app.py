import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch

# --- Model Loading (same as step 3) ---
model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Image Quality Check Function (same as step 4) ---
def check_image_quality(img_file):
    image = Image.open(img_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)

    if predicted.item() % 3 == 0:
        return "High Quality ✅"
    elif predicted.item() % 3 == 1:
        return "Needs Improvement ⚠️"
    else:
        return "Low Quality ❌"

# --- Streamlit App Layout ---
st.set_page_config(page_title="AdImage Classifier", page_icon="🖼️")
st.title("🖼️ AdImage Classifier - AI Ad Visual Quality Checker")
st.write("Upload an ad image (JPG/PNG) to instantly check its quality!")

uploaded = st.file_uploader("Upload an Ad Image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    st.write("---") # Separator

    # Add a spinner while processing
    with st.spinner('Analyzing image quality...'):
        result = check_image_quality(uploaded)
    
    # Display the prediction with an appropriate icon
    if "High Quality" in result:
        st.success(f"🔍 Prediction: {result}")
    elif "Needs Improvement" in result:
        st.warning(f"🔍 Prediction: {result}")
    else:
        st.error(f"🔍 Prediction: {result}")

    st.info("💡 Note: This is a demonstration using a simulated quality check. For production use, a fine-tuned model is recommended.")
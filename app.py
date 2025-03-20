import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision import models

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
model.eval()

# Streamlit app
st.title("🕵️‍♂️ Deep Fake Face Detector")
st.write("Upload an image, and the AI will detect if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][1].item()
        predicted = torch.argmax(probabilities, 1).item()

    label = 'Fake' if predicted == 1 else 'Real'
    st.write(f"**Prediction:** {label} ({confidence*100:.2f}% Fake Confidence)")

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# =====================================================
# MODEL
# =====================================================

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# LOAD MODEL
# =====================================================

classes = torch.load("classes.pth")

model = SmallCNN(num_classes=len(classes))
model.load_state_dict(torch.load("best_word_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("My Handwriting Word Recognizer")
st.write("Upload a handwritten word image")

uploaded = st.file_uploader(
    "Choose image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    st.success(f"Prediction: {classes[pred]}")
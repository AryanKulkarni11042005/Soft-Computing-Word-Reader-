import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# =====================================================
# CHARACTER VOCAB
# =====================================================

characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-\"():;#&/ "
char_to_idx = {c: i + 1 for i, c in enumerate(characters)}
idx_to_char = {i + 1: c for i, c in enumerate(characters)}
BLANK_IDX = 0

# =====================================================
# MODEL
# =====================================================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        self.rnn = nn.LSTM(
            input_size=256 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = CRNN(num_classes=len(char_to_idx) + 1)
    model.load_state_dict(torch.load("crnn_word_reader_final.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):
    image = image.convert('L')

    new_w = int(image.size[0] * (32 / image.size[1]))
    image = image.resize((new_w, 32))

    if new_w < 128:
        image = ImageOps.expand(image, border=(0, 0, 128 - new_w, 0), fill=255)
    else:
        image = image.crop((0, 0, 128, 32))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image

# =====================================================
# DECODER
# =====================================================

def decode_prediction(output):
    output = output.argmax(2)
    output = output[:, 0]

    result = []
    prev = -1

    for p in output:
        p = p.item()

        if p != prev and p != BLANK_IDX:
            result.append(idx_to_char.get(p, ''))

        prev = p

    return ''.join(result)

# =====================================================
# UI
# =====================================================

st.title("Handwritten Word Recognition")
st.write("Upload a cropped handwritten word image")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)

    with torch.no_grad():
        output = model(processed)
        output = output.permute(1, 0, 2)

        prediction = decode_prediction(output)

    st.subheader("Prediction")
    st.success(prediction)
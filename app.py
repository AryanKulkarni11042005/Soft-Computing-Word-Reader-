import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Rebuild Vocab
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
int_to_char = {i + 1: char for i, char in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1

# 1. Model Definition
class HandwrittenWordReader(nn.Module):
    def __init__(self, num_chars, hidden_size=256, num_layers=2):
        super(HandwrittenWordReader, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )
        self.rnn = nn.LSTM(256 * 4, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        conv_out = conv_out.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv_out)
        return self.fc(rnn_out) # No log_softmax needed for greedy decoding

# 2. UI Setup
st.set_page_config(page_title="Word Reader", page_icon="📝")
st.title("📝 Handwritten Word Recognition")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenWordReader(num_chars=NUM_CLASSES)
    model.load_state_dict(torch.load('crnn_word_reader.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. Inference
uploaded_file = st.file_uploader("Upload a cropped WORD image...", type=["jpg", "png"])

def decode_predictions(preds):
    # Greedy decoding: pick the highest probability class at each timestep
    _, max_indices = torch.max(preds, dim=-1)
    max_indices = max_indices.squeeze(0).cpu().numpy()
    
    # Remove blanks (0) and consecutive duplicates (CTC rules)
    decoded_text = []
    prev_char = None
    for idx in max_indices:
        if idx != 0 and idx != prev_char:
            decoded_text.append(int_to_char[idx])
        prev_char = idx
    return "".join(decoded_text)

if uploaded_file is not None:
    im = Image.open(uploaded_file).convert('L')
    st.image(im, caption="Uploaded Word", width=300)
    
    if st.button("Read Word"):
        cur_width, cur_height = im.size
        new_width = int(cur_width * (32 / cur_height))
        im = im.resize((new_width, 32), Image.LANCZOS)
        
        if new_width < 128:
            im = ImageOps.expand(im, (0, 0, 128 - new_width, 0), fill=255)
        else:
            im = im.crop((0, 0, 128, 32))
            
        img_tensor = transforms.ToTensor()(im).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(img_tensor)
            result = decode_predictions(preds)
            
        st.success(f"### 🎯 Predicted Text: **{result}**")
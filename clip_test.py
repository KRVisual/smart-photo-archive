import clip
import torch
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load one of your photos
image = preprocess(Image.open("photos/IMG_0269.JPG")).unsqueeze(0).to(device)

# Descriptions to test against
descriptions = [
    "cherry blossoms",
    "people walking on a street",
    "a building",
    "a cat",
    "sunset over water"
]

# Tokenize the descriptions
text = clip.tokenize(descriptions).to(device)

# Compare image to descriptions
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("How well each description matches the photo:\n")
for desc, prob in zip(descriptions, probs[0]):
    print(f"{desc}: {prob:.1%}")
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip().lower()

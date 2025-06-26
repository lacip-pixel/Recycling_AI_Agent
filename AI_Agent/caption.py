# caption.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load BLIP-2 model and processor
# This model requires a GPU with at least ~6-7 GB of VRAM
try:
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        torch_dtype=torch.float16
    )
except Exception as e:
    raise RuntimeError("Failed to load BLIP-2 model. Ensure you have a compatible GPU and sufficient memory.") from e

def caption_image(image_path: str) -> str:
    """
    Generate a natural language caption for an image using BLIP-2.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Caption describing the image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"error: unable to open image - {e}"

    try:
        # Preprocess image and generate caption
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip().lower()

    except Exception as e:
        return f"error: caption generation failed - {e}"

# caption.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Use a smaller, more practical model for local development
MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_caption_model():
    """
    Lazy-load the image captioning model so it doesn't load at import time.
    This prevents Flask debug reload from repeatedly loading a huge model.
    """
    global _processor, _model

    if _processor is None or _model is None:
        try:
            _processor = BlipProcessor.from_pretrained(MODEL_NAME)
            _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
            _model.to(_device)
            _model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load caption model '{MODEL_NAME}'. Original error: {e}"
            ) from e

    return _processor, _model


def caption_image(image_path: str) -> str:
    """
    Generate a natural language caption for an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Caption describing the image, or an error string.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"error: unable to open image - {e}"

    try:
        processor, model = load_caption_model()

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=4
            )

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip().lower()

    except Exception as e:
        return f"error: caption generation failed - {e}"
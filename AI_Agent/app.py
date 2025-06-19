# app.py
from flask import Flask, render_template, request
from ai_agent import classify_item
from caption import caption_image  # For BLIP image captioning
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = None
    past_history = []

    if request.method == 'POST':
        item = request.form.get('item')
        location = request.form.get('location')
        home_composting = "yes" if request.form.get("home_composting") else "no"
        image = request.files.get('image')

        if not location:
            message = "❗ Please provide your location."
        elif image and image.filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            caption = caption_image(image_path)
            result = classify_item(caption, location, home_composting)
            message = f"🖼️ Image identified as: '{caption}'"

        elif item:
            result = classify_item(item, location, home_composting)

        else:
            message = "❗ Please provide either a description or an image."

    return render_template("index.html", result=result, message=message, past_history=past_history)

if __name__ == '__main__':
    app.run(debug=True)

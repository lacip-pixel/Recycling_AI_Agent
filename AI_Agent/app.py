from flask import Flask, render_template, request
from ai_agent import classify_item, memory
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
        image = request.files.get('image')

        # Placeholder for image classification
        if image and image.filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            message = "🖼️ Image classification coming soon!"

        elif item and location:
            result = classify_item(item, location)

    # Show past conversation memory (human and AI messages)
    for msg in memory.chat_memory.messages:
        if hasattr(msg, "type") and msg.type in ["human", "ai"]:
            past_history.append((msg.type.capitalize(), msg.content))

    return render_template("index.html", result=result, message=message, past_history=past_history)

if __name__ == '__main__':
    app.run(debug=True)

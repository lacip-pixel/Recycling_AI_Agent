# app.py
from flask import Flask, render_template, request
from ai_agent import classify_item, memory  # import memory
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = None
    history = None

    if request.method == 'POST':
        item = request.form.get('item')
        location = request.form.get('location')
        image = request.files.get('image')

        if image and image.filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            message = "Image classification coming soon!"

        elif item:
            result = classify_item(item, location)
            history = memory.load_memory_variables({})["history"]

    return render_template("index.html", result=result, message=message, history=history)

if __name__ == '__main__':
    app.run(debug=True)

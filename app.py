from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
model = load_model("mnist_model.h5")

def preprocess(img_base64):
    img = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("L").resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img = preprocess(data["image"])
    prediction = model.predict(img)
    return jsonify({"prediction": int(np.argmax(prediction))})

@app.route("/")
def home():
    return "CNN Digit Recognizer is live!"

if __name__ == "__main__":
    app.run(debug=True)

from io import BytesIO
from PIL import Image
from inference import predict
from app import app
from flask import Flask, request 
import json

@app.route("/predicts/", methods=['POST'])
def submit_file():
    if request.method == 'POST':
        file = request.files['file']
        annotations = json.load(request.files['data'])['annotations']
        image = Image.open(BytesIO(file.read()))
        count = predict(image, annotations)
        return {'count': count} 

if __name__ == "__main__":
    app.run(host='0.0.0.0')


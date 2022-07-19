from io import BytesIO
from PIL import Image
from inference import predict
from app import app
from flask import Flask, request 
import json

@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['file']
        annotations = json.load(request.files['data'])['annotations']
        image = Image.open(BytesIO(file.read()))
        count = predict(image, annotations)
        return {'count': count} 

@app.route("/heatmap", methods=['POST'])
def evaluate_heatmap():
    if request.method == 'POST':
        file = request.files['file']
        annotations = json.load(request.files['data'])['annotations']
        image = Image.open(BytesIO(file.read()))
        count, heatmap = predict(image, annotations, return_density_map=True)
        return {'count': count, 'heatmap': heatmap} 

if __name__ == "__main__":
    app.run(host='0.0.0.0')


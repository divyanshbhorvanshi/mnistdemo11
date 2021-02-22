import numpy as np
import torch
from flask import Flask, request, jsonify
import json
import torch.nn as nn
from torchvision import transforms
from app.torch_utils import Net

app = Flask(__name__)
normal = transforms.Normalize(torch.tensor([0.1307]), torch.tensor([0.3081]))
model = Net()
model.load_state_dict(torch.load('app/mnist.pth'))
model.eval()

@app.route("/predict", methods=["POST"])

def predict():
    meta = json.load(request.files['meta'])
    img = request.files['img'].read()
    im = np.frombuffer(img, dtype=np.uint8).astype('float32')
    im = torch.from_numpy(im).view(-1, *meta['shape'])
    transform = transforms.Compose([
        transforms.Resize((28,28), interpolation=2),
        normal
    ])
    im = transform(im)
    output = model(im.unsqueeze(0))
    _, predicted = torch.max(output, dim = 1)
    pred = torch.sort(output,dim = 1, descending = True)[1].tolist()[0][:]
    print(pred)
    return jsonify(int(predicted))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

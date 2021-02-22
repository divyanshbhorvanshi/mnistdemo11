import requests
import json
import io
import sys
import numpy as np
from PIL import Image, ImageOps

im = Image.open(sys.argv[1])
im = ImageOps.grayscale(im)
im = np.array(im)
meta = io.StringIO(json.dumps({'shape': list(im.shape)}))
data = io.BytesIO(bytearray(im))
r = requests.post('https://mnistdemo2.herokuapp.com/predict',
                  files={'meta': meta, 'img' : data})
response = json.loads(r.content)

print("Predicted Class:", response)

import io
import json

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('class_index.json')) # class id to name mapping 
model = models.resnet18(pretrained=True) # define our model here
model.load_state_dict(torch.load('birb.pt')) # load the saved model state here
model.eval()

def transform_image(image_bytes):  # modify the transform function to suit our model
    my_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# test the transform function
# with open("../_static/img/sample_file.jpeg", 'rb') as f:
#     image_bytes = f.read()
#     tensor = transform_image(image_bytes=image_bytes)
#     print(tensor)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == "__main__":
    app.run(debug=True)


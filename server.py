import base64
import flask
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from io import BytesIO
import requests

# Initialize the model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device='cuda')
predictor = SamPredictor(sam)

app = flask.Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    data = flask.request.json
    image_data = data.get('url')
    if not image_data:
        return flask.jsonify({"error": "No data provided"}), 400

    try:
        # Check if the URL starts with HTTP, fetch the data if true
        if image_data.startswith("http"):
            response = requests.get(image_data)
            if response.status_code == 200:
                image_bytes = response.content
            else:
                return flask.jsonify({"error": "Failed to fetch image from URL"}), 500
        else:
            # Decode the base64 string directly if it's not a URL
            image_bytes = base64.b64decode(image_data)

        # Convert bytes to an image and handle the conversion to RGB
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return flask.jsonify({"error": "Failed at image conversion: " + str(e)}), 500

    try:
        # Convert the image to a numpy array and process it
        image_np = np.array(image)
        predictor.set_image(image_np)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        image_embedding = image_embedding.tobytes()

        # Encode the numpy bytes to base64 and return
        base64_image_embedding = base64.b64encode(image_embedding).decode('ascii')
        return flask.jsonify({"image_embedding_base64": base64_image_embedding})
    except Exception as e:
        return flask.jsonify({"error": "Failed during processing: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

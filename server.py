from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from io import BytesIO

# Initialize the model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device='cuda')
predictor = SamPredictor(sam)

app = flask.Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    data = flask.request.json
    base64_data = data.get('url')
    if not base64_data:
        return flask.jsonify({"error": "No data provided"}), 400

    try:
        # Check and strip the prefix if it exists
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]

        # Decode the base64 string
        image_bytes = base64.b64decode(base64_data)

        # Convert bytes to an image and handle the conversion to RGB
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return flask.jsonify({"error": "Failed at image conversion: " + str(e)}), 500

    try:
        # Convert the image to a numpy array and process it
        image_np = np.array(image)
        predictor.set_image(image_np)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        print(image_embedding.shape)

        image_embedding = image_embedding.tobytes()

        # Encode the numpy bytes to base64 and return
        base64_image_embedding = base64.b64encode(image_embedding).decode('ascii')
        # print("embedding:", base64_image_embedding)
        return flask.jsonify({"image_embedding_base64": base64_image_embedding})
    except Exception as e:
        return flask.jsonify({"error": "Failed during processing: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
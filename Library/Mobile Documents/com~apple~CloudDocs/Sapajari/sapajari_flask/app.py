from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import *
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model



# Initialize Flask
app = Flask(__name__)
CORS(app)

# ------------ MASTER --------------
model = load_model('model_sapajari.h5')

@app.route('/', methods=['GET'])
def hello():
    response = {
        'status': True,
        'message': 'Hi there! Server is running!',
        'data': None
    }
    return jsonify(response), 200
# SCAN NUTRITION
@app.route('/predict', methods=['POST'])
def predict():
    classes = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
    }
    #Loag image
    image = request.files['image']
    (wristX, wristY, wristZ,
     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
     thumb_McpX, thumb_McpY, thumb_McpZ,
     thumb_IpX, thumb_IpY, thumb_IpZ,
     thumb_TipX, thumb_TipY, thumb_TipZ,
     index_McpX, index_McpY, index_McpZ,
     index_PipX, index_PipY, index_PipZ,
     index_DipX, index_DipY, index_DipZ,
     index_TipX, index_TipY, index_TipZ,
     middle_McpX, middle_McpY, middle_McpZ,
     middle_PipX, middle_PipY, middle_PipZ,
     middle_DipX, middle_DipY, middle_DipZ,
     middle_TipX, middle_TipY, middle_TipZ,
     ring_McpX, ring_McpY, ring_McpZ,
     ring_PipX, ring_PipY, ring_PipZ,
     ring_DipX, ring_DipY, ring_DipZ,
     ring_TipX, ring_TipY, ring_TipZ,
     pinky_McpX, pinky_McpY, pinky_McpZ,
     pinky_PipX, pinky_PipY, pinky_PipZ,
     pinky_DipX, pinky_DipY, pinky_DipZ,
     pinky_TipX, pinky_TipY, pinky_TipZ,
     output_IMG) = extract_feature(image)
    input_IMG = np.array([[[wristX], [wristY], [wristZ],
                     [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                     [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                     [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                     [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                     [index_McpX], [index_McpY], [index_McpZ],
                     [index_PipX], [index_PipY], [index_PipZ],
                     [index_DipX], [index_DipY], [index_DipZ],
                     [index_TipX], [index_TipY], [index_TipZ],
                     [middle_McpX], [middle_McpY], [middle_McpZ],
                     [middle_PipX], [middle_PipY], [middle_PipZ],
                     [middle_DipX], [middle_DipY], [middle_DipZ],
                     [middle_TipX], [middle_TipY], [middle_TipZ],
                     [ring_McpX], [ring_McpY], [ring_McpZ],
                     [ring_PipX], [ring_PipY], [ring_PipZ],
                     [ring_DipX], [ring_DipY], [ring_DipZ],
                     [ring_TipX], [ring_TipY], [ring_TipZ],
                     [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                     [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                     [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                     [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])
    predictions = model.predict(input_IMG)
    predicted_class_index = predictions.argmax()
    predicted_alphabet = classes[predicted_class_index]

    if type(output_IMG) is not np.ndarray:
        # 404: Not Found
        response = {
            'data': None,
            'message': 'Not Detected!',
            'status': False,
        }
    else:
        # 200: Success
        response = {
            'data': predicted_alphabet,
            'message': 'Detected!',
            'status': True,
        }
    return jsonify(response), 200


# Initialize Flask
app.debug = True
CORS(app)

if __name__ == '__main__':
    app.run()

import keras
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import initializers
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from flask import request
from flask import jsonify
from flask import Flask
import h5py
import math

app = Flask(__name__)

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    size=(224,224)
    image = image.resize(size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print("LOADING MODEL......")
global graph
graph = tf.get_default_graph()
global model
tf.keras.models.load_model('model.h5')
print("MODEL LOADED!!!")

@app.route("/predict", methods=["POST"])
def doPrediction():
    print("GOT REQUEST")
    message = request.get_json(force=True)
    #print(message)
    response = message['image']
    encoded = response[23:]
    print(encoded)
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    image = Image.open(dataBytesIO)
    processed_image = preprocess_image(image)     
    graph = tf.get_default_graph()        
    with graph.as_default():
        prediction = model.predict(processed_image)
#     keras.backend.clear_session()
    print(prediction[0])
    response = {
        'predictions': {
            'G0' : int(np.around(prediction[0][0])),
            'G1' : int(np.around(prediction[0][1])),
            'G2' : int(np.around(prediction[0][2])),
            'G3' : int(np.around(prediction[0][3])),
           # 'G4' : int(np.around(prediction[0][4]))    
        }   
    } 
    return jsonify(response)

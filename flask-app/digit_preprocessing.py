import tensorflow as tf
import cv2, base64
import numpy as np


#loading the model
model = tf.keras.models.load_model("C:/Users/AKUMA/Desktop/Repository/digit-recognition/mnist_digit_recog_cnn_2-conv-128-nodes-1-dense-17-04-2020_00-09_Ver16.h5")


def base64_to_nparray(URL):
    # https://gist.github.com/daino3/b671b2d171b3948692887e4c484caf47#gistcomment-3163871
    image_b64 = URL.split(",")[1]
    binary = base64.b64decode(image_b64)
    image = np.asarray(bytearray(binary), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) 


def analyze_canvas(canvas_data_URL):

    # convert base64 canvas image, to a numpy array
    array = base64_to_nparray(canvas_data_URL)
    
    # resize to 28x28 px
    resized = cv2.resize(array,(28,28))
    
    # reshape to fit model input layer
    reshaped = resized.reshape(1,28,28,1)

    # predict
    softmax_prediction = model.predict(reshaped)
    argmax_prediction = np.argmax(softmax_prediction)

    prediction = {
        "softmax": softmax_prediction.tolist(),
        "argmax": int(argmax_prediction),
    }

    print(prediction)

    return prediction
     
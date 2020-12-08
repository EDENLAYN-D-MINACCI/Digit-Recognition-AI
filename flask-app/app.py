from flask import Flask, render_template, Response, request, make_response, jsonify
from digit_preprocessing import analyze_canvas
import cv2
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)  # or __name__ for current module
logger.setLevel(logging.DEBUG)
camera = cv2.VideoCapture(0)

@app.route('/print')
def printMsg():
    app.logger.warning('testing warning log')
    app.logger.error('testing error log')
    app.logger.info('testing info log')
    return "Check your console"

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/canvas', methods=['POST', 'GET'])
def canvas():

    prediction = {}
    
    if request.method == "POST":
        canvas_URL = request.get_json()
        prediction = jsonify(analyze_canvas(canvas_URL))
        return make_response(prediction, 200)

    return render_template('canvas.html')
    


@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__ == '__main__':
    # defining server ip address and port
    app.run(debug=True)



def get_frame(self):
    #extracting frames
    ret, frame = self.video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # encode OpenCV raw frame to jpg and displaying it
    ret, jpeg = cv2.imencode('.jpg', gray)
    return jpeg.tobytes()
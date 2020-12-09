from flask import Flask, render_template, Response, request, make_response, jsonify
from digit_preprocessing import analyze_canvas, analyze_frame
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)  # or __name__ for current module
logger.setLevel(logging.DEBUG)


@app.route('/canvas', methods=['POST', 'GET'])
def canvas():

    prediction = {}
    
    if request.method == "POST":
        canvas_URL = request.get_json()
        prediction = jsonify(analyze_canvas(canvas_URL))
        return make_response(prediction, 200)

    return render_template('canvas.html')
    

@app.route('/stream', methods=['POST', 'GET'])
def stream():
    preprocessing_stage = "default"

    if request.method == "POST":
        preprocessing_stage = request.get_json()
    
    return Response(analyze_frame(preprocessing_stage), mimetype='multipart/x-mixed-replace; boundary=frame')

    
@app.route('/stream-parameter')
def stream_template():
    return render_template('stream.html')

    




if __name__ == '__main__':
    # defining server ip address and port
    app.run(debug=True)




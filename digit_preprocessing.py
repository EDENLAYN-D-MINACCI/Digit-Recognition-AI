from tensorflow.keras.models import load_model
import cv2, base64
import numpy as np


#loading the model
model = load_model("C:/Users/AKUMA/Desktop/Repository/digit-recognition/mnist_digit_recog_cnn_2-conv-128-nodes-1-dense-17-04-2020_00-09_Ver16.h5")


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
     


def analyze_frame(preprocessing_stage):
    # how to read box coordinates ; https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return/51952289

    VideoCapture = cv2.VideoCapture(0)
    color = (255,255,255)
    thickness = 2

    def empty():
        pass
        
    windowName = "parameters"
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 800,550)

    
    cv2.createTrackbar("pixel_neighboors", windowName, 73, 500, empty) 
    cv2.createTrackbar("Constant", windowName, 4, 500, empty) 
    cv2.createTrackbar("blur", windowName, 8, 100, empty) 
    cv2.createTrackbar("area min", windowName, 327, 10000, empty)
    cv2.createTrackbar("area max", windowName, 2675, 10000, empty)
    cv2.createTrackbar("rectangle length min", windowName, 250, 10000, empty)
    cv2.createTrackbar("rectangle length max", windowName, 700, 10000, empty)
    cv2.createTrackbar("x length", windowName, 457, 1000, empty)
    cv2.createTrackbar("y length", windowName, 388, 1000, empty)
    
        
        
    while True:
    #__________________________________________________________________________________________________________________                        
    # setting up parameters

        # storing trackbar value in variables
        pixel_neighboors_value  = cv2.getTrackbarPos("pixel_neighboors", windowName)
        Constant_value          = cv2.getTrackbarPos("Constant", windowName)
        blur_value              = cv2.getTrackbarPos("blur", windowName)
        threshold_area_min      = cv2.getTrackbarPos("area min", windowName)
        threshold_area_max      = cv2.getTrackbarPos("area max", windowName)
        rect_length_min         = cv2.getTrackbarPos("rectangle length min", windowName)
        rect_length_max         = cv2.getTrackbarPos("rectangle length max", windowName)
        threshold_x_length      = cv2.getTrackbarPos("x length", windowName)
        threshold_y_length      = cv2.getTrackbarPos("y length", windowName)
        
        
        # need those conditions otherwise some variables will raise an error
        if blur_value < 2:
            blur_value = 2
            
        if  (pixel_neighboors_value % 2) == 0:
            pixel_neighboors_value = 7
            
        elif pixel_neighboors_value < 1:
            pixel_neighboors_value = 7

        # recognized digit (and false positives) will be stocked here, it will be emptied at the beginning of each frame
        preprocessed_digits = []
        
        
        
    #__________________________________________________________________________________________________________________                        
    # beginning image processing


        # reading video feed
        ret, frame = VideoCapture.read()
        print(frame.shape)
        
        # defining width and height of the camera
        frame_width, frame_height, frame_channel = frame.shape
                                
        #setting up screen limits for bounding box detections 
        start_point_1 = (int(frame_width/2 + frame_width/2), 0)
        end_point_1   = (int(frame_width/2 + frame_width/2), frame_height)

        start_point_2 = (int(frame_width/4), 0)
        end_point_2   = (int(frame_width/4), frame_height)

        #cv2.line(frame, start_point_1, end_point_1, color, thickness)                     
        #cv2.line(frame, start_point_2, end_point_2, color, thickness)                     

        
        # frame converted to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame blured by default kernel size 8
        blur = cv2.blur(gray,(blur_value,blur_value))
        
        
        # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        # maxValue is the pixel value give to the pixel above the treshold
        # block Size is the number of neighboring pixels, it makes binary object thicker (it only accepts odd values)    
        # C is a constant that actually acts like a noise reductor
        # this type of threshold acts better with variable light source
        adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, pixel_neighboors_value, Constant_value)

        # cv2.findContours(image, mode, method, contours, hierarchy, offset) 
        # mode is the way of finding contours, and method is the approximation method for the detection. 
        contours, hierachy = cv2.findContours(adaptive_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )


    #__________________________________________________________________________________________________________________                        
    # getting contours coordinates

        for points in contours:
            
            # obtains the 4 corner points of the contour rectangle.    
            x,y,w,h = cv2.boundingRect(points) 
            
            # area is x length times y lengths)
            area = int(cv2.contourArea(points))
            #cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            
            # rectangle length is the length of a closed rectangle
            rectangle_length = int(cv2.arcLength(points, True))
            #cv2.putText(frame, str(rectangle_length), (x + 20 ,y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
            
            
            
            
    #__________________________________________________________________________________________________________________                        
    # set threshold conditions for filter the detected rectangles

            if area > threshold_area_min and area < threshold_area_max:
                if rectangle_length > rect_length_min and rectangle_length < rect_length_max:
                    if x + w < threshold_x_length and y + h < threshold_y_length:
                        if x < int(frame_width/2 + frame_width/2) and x > int(frame_width/4):
                    
                    
                    
                            # digit pre-processing
                            # Cropping out the digit from the thresholding image
                            ROI = adaptive_threshold[y:y+h, x:x+w]    

                            # Resizing the digit to (18, 18)
                            resized_digit = cv2.resize(ROI, (18,18))   

                            # Padding the digit with 5 pixels of black color (zeros) in each side to produce the image of (28, 28)
                            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

                            # Adding the preprocessed digit to the list of preprocessed digits
                            preprocessed_digits.append(padded_digit)

                            #setting coordinates                    
                            start = (x, y)
                            end = (w + x, h + y)

                            #drawing bounding box around detected objects
                            cv2.rectangle(frame, start, end, color, thickness)

                            
                            #cv2.drawContours(frame, contours, contourIdx = -1, color = (255, 0, 0), thickness = 2)
                            
                            
    #__________________________________________________________________________________________________________________                        
    # make the prediction


                            for digit in preprocessed_digits:
                                prediction = model.predict(digit.reshape(1, 28, 28, 1))  

                            debug_prediction = "Final Output: {}".format(np.argmax(prediction)) 
                            print(debug_prediction)
                            cv2.putText(frame, debug_prediction, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)



    #__________________________________________________________________________________________________________________                        
    # Debug windows
            
    
        #cv2.imshow("original", frame)
        #cv2.imshow("gray", gray)
        #cv2.imshow("blur", blur)
        #cv2.imshow("ROI", ROI)
        #cv2.imshow("resized", resized_digit)
        #cv2.imshow('thres', adaptive_threshold)
        #cv2.imshow("padded", padded_digit)
        print(preprocessing_stage)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


        if cv2.waitKey(1) == ord('q'):
            break
            
    VideoCapture.release()
    cv2.destroyAllWindows()


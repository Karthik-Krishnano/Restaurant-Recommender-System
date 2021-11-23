from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
#import face_recognition
import keras
from keras.models import load_model
import cv2

################face recognition###############################

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

#image = cv2.imread("4.jpg")
#gray = cv2.imread("1.jpg", cv2.COLOR_RGB2GRAY)

cap=cv2.VideoCapture(0)

while True:

    ret,image=cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    x,y,h,w=50,50,50,50

    try:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #print(faces)
        
    except:
        print("NO")

    #cropped_image = img[80:280, 150:330]
    crop_img = image[y:y+h, x:x+w]

    #Emotion Detection
    emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

    # resizing the image
    face_image = cv2.resize(crop_img, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    model = load_model("emotion_detector_models/model_v6_23.hdf5")

    predicted_class = np.argmax(model.predict(face_image))

    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    predicted_label = label_map[predicted_class]

    print(predicted_label)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (x,y)
    
    # fontScale
    fontScale = 2
    
    # Blue color in BGR
    color = (0, 255, 0)
    
    # Line thickness of 2 px
    thickness = 4

    # Using cv2.putText() method
    image = cv2.putText(image, predicted_label, org, font,fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Faces found", image)
    #cv2.imshow("Faces", crop_img)
    cv2.waitKey(2)
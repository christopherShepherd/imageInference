#portrait_1  Christopher Shepherd 2017

#import dependencies
import numpy as np
import cv2

#detector class to detect faces
from detector import Detector

#predictor class to infer from detected faces
from predictor import Predictor

#load starting image
titleImage = cv2.imread('images/titleScreen.png')

#set up window context
cv2.namedWindow('portrait_1', cv2.WINDOW_NORMAL)
#set window to Fullscreen
cv2.setWindowProperty('portrait_1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

windowName = 'portrait_1'

programOn = True
#while program is running, close program with 'q' (detector run returns false)
while programOn:

    #new detector
    det = Detector(titleImage, windowName)

    #search for faces
    detectedFace = det.run()

    #if the detector does not retrun an image, close the program
    if not isinstance(detectedFace, np.ndarray):
        programOn = False

    else:
        #if the detector returns an image

        #create a predictor
        predictor = Predictor(detectedFace, windowName)
        #predictor runs until it has finished the inference calculation and display cycle
        finalImage = predictor.predict()

        if not isinstance(finalImage, np.ndarray):
            programOn = False
        else:
            #once the predictor is finished, the final image it output is passed back
            #to the detector so that the process can start over again
            titleImage = finalImage

cv2.destroyAllWindows()




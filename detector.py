#A class to detect faces viewing the work and select one for Generator processing

#import dependencies
import cv2
import numpy as np
from timeit import default_timer as timer
import math
import random

class Detector:

    #intiialiser takes a starting frame and window name
    def __init__(self, startFrame, windowName):
        self.initialFrame = np.array(startFrame)
        self.initialFrame = cv2.resize(self.initialFrame, (480, 480), interpolation = cv2.INTER_CUBIC)
        self.windowName = windowName
        
        #select desired video capture, set to 0 if default webcam is desired
        self.cap = cv2.VideoCapture(0)

        #manage different stages of Detector
        self.detecting = True
        self.startTransition = True
        self.endTransition = False

        #for face detection
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        #for timer
        self.timing = False
        self.startTime = timer()
        self.endTime = timer()

        #for the end transition 
        self.firstMove = True
        self.scaleValue = 0
        
        #chosenFace variable holds the frame to be passed to the generator
        #xPos holds top-left corner x-position for chosenFace within capture frame
        #yPos holds top-left corner y-position for chosenFace within capture frame
        #dimension holds the width and height of chosenFace selection
        self.xPos = 0
        self.yPos = 0
        self.dimension = 0
        self.chosenFace = None



    def run(self):

        #for start transition, chance that a pixel from startFrame is displayed
        chance = 1.0
        
        #black canvas to cover the whole screen
        outputFrame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        #testerImage = cv2.imread('mean.png')
        #test = cv2.resize(testerImage, (480, 480))

        #while detecting(a valid face has not yet been found)
        while self.detecting:

            #read video capture
            ret, frame  = self.cap.read()

            ########################################################################
            #--if transitioning from static frame to cam capture
            if self.startTransition:
               
                ##pixilated cross between two images
                # chance that a pixel from the static image is displayed diminishes over time

                for numx in range(0, 480):
                    for numy in range(0, 480):

                        check = random.random()

                        #if random value is less than current threshold take the current
                        #pixel from the initial frame
                        if check < chance:
                            frame[numx, numy + 80] = self.initialFrame[numx, numy]


                #decrease chance that pixels are chosen
                chance -= 0.02

                #if pixels are barely selected, transition is over
                if chance < 0.05:
                    self.startTransition = False


            ######################################################################
            #--transitioning from web capture to chosen face
            elif self.endTransition:

                #if first move, move towards the centre without expanding,
                #once in the centre expand and move xPos yPos towards 0, 0

                
                #find values to scale and move the chosenFace
                directionX = 80 - self.xPos
                directionY = 0 - self.yPos

                if  directionX >= -25 and directionX < 0:
                    self.xPos -= 1
                elif directionX > 0 and directionX <=25:
                    self.xPos += 1
                else:
                    self.xPos += int(directionX/25)

                if  directionY >= -25 and directionY < 0:
                    self.yPos -= 1
                elif directionY > 0 and directionY <= 25:
                    self.yPos += 1
                else:
                    self.yPos += int(directionY/25)


                scaleDiff = frame.shape[0] - self.scaleValue
                scaleDiff = int(scaleDiff/20)
                if scaleDiff < 1 :
                    scaleDiff = 1

                #first resize according to resize factor
                holdingImage = cv2.resize(self.chosenFace, (self.scaleValue+5, self.scaleValue+5))
                
                #create x and y padding values as necessary(selection extends beyond the frame edge
                xLow = 0
                yLow = 0 
                xHigh = self.scaleValue 
                yHigh = self.scaleValue

                if self.xPos < 0:
                    xLow = abs(self.xPos)

                if self.yPos < 0:
                    yLow = abs(self.yPos)

                if (self.xPos + self.scaleValue) > frame.shape[1]:
                    xHigh = self.scaleValue - ((self.xPos + self.scaleValue)-frame.shape[1])
                if (self.yPos + self.scaleValue) > frame.shape[0]:
                    yHigh = self.scaleValue - ((self.yPos + self.scaleValue)-frame.shape[0])

                #copy the chosen face into the current frame
                for i in range(yLow, yHigh):
                    for j in range(xLow, xHigh):
                        frame[self.yPos + i, self.xPos + j] = holdingImage[i, j]
                        #self.chosenFace[i, j]

                if self.scaleValue+scaleDiff <= frame.shape[0]:
                    self.scaleValue += scaleDiff 


                #if the detected face image is the size of the window height, the end transition is finished
                if holdingImage.shape[0] < (frame.shape[0] + 20) and holdingImage.shape[0] > (frame.shape[0]-10):# and abs(self.xPos <2) and abs(self.yPos < 2):
                    self.cap.release()
                    return self.chosenFace
                    #once transition finished, release the video capture and return the
                    #chosenFace image for prediction



            ######################################################################
            #--searching for valid faces to predict from
            # Adapted from the camera and crop programs of Part7 in this project
            #For a face to be chosen, it must be detected for at least 2seconds before
            # a further validation and eventual face selection can occur
            else:
                #face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

                #face detection timer to check that people are there
                #before starting the inference
                if len(faces) > 0:

                    #if face detected and no timer running start the timer
                    if not self.timing:
                        self.timing = True
                        self.startTime = timer()
                    else:
                        elapsedTime = timer() - self.startTime
                        print('timing! Elapsed: ', elapsedTime)

                        #if face detected and timer past required time, check for validation to 
                        #begin the prediction process
                        if elapsedTime > 2.0:

                            #->(if face detected but no eyes perhaps consider increasing the image size???)

                            #face and eye detection
                            for (x, y, w, h) in faces:
                                
                                roi_gray = gray[y:y+h, x:x+w]
                                eyes = self.eyeCascade.detectMultiScale(roi_gray, 1.3, 5)

                                if len(eyes) > 1 and (eyes[1][0]-eyes[0][0]) > 15:

                                    #dimensions for the chosenFace selection are determined to be
                                    #7 times the height of the first eye detected
                                    cropHeight = int(eyes[0][3] * 7)
                                    cropWidth = cropHeight

                                    #top of chosenFace selection is 3.1 x eyeheight above 
                                    #the eye detection area. This ensures eyes are always centre 
                                    #of the selection box
                                    topCorner = eyes[0][1] - int(eyes[0][3] * 3.1)
                                    #left edge of selection box is mid way between the two eyes
                                    #minus half of the width/height of the selection
                                    leftCorner = int((eyes[0][0] + (eyes[1][2] + eyes[1][0]))/2) - int(cropHeight/2)


                                    #eyeHeight compared to face detection size
                                    faceEyeFraction = h/eyes[0][3]
                                    #width of face detection compared to above calculated crop width
                                    cropFaceFraction = cropWidth/w

                                    #making sure the face is within the valid range for prediction
                                    if 1.59 < cropFaceFraction < 1.91 and 3.49 < faceEyeFraction < 4.51:

                                        self.xPos = x + leftCorner
                                        self.yPos = y + topCorner
                                        self.dimension = cropHeight

                                        self.scaleValue = cropHeight

                                        #create the chosen frame, if it is beyond the boundaries 
                                        #of the frame then will transfer necessary values to a 
                                        #zero-filled array of correct dimensions
                                        print(frame.shape[0], ' x ', frame.shape[1], 'chosenDim', self.dimension)
                                        #if chosen selection is beyond the frame boundaries
                                        if self.yPos < 0 or self.xPos < 0 or (self.yPos + self.dimension) > frame.shape[0] or (self.xPos + self.dimension) > frame.shape[1]:

                                            #zeros of size chosen selection dimension
                                            self.chosenFace = np.zeros((self.dimension, self.dimension, 3), dtype=np.uint8)

                                            #default values for filling zeros array
                                            xLow = 0
                                            yLow = 0
                                            xHigh = self.dimension 
                                            yHigh = self.dimension 

                                            #create x and y padding values as necessary
                                            if self.xPos < 0:
                                                xLow = abs(self.xPos)

                                            if self.yPos < 0:
                                                yLow = abs(self.yPos)

                                            if (self.xPos + self.dimension) > frame.shape[1]:
                                                xHigh = self.dimension - ((self.xPos + self.dimension)-frame.shape[1])
                                            if (self.yPos + self.dimension) > frame.shape[0]:
                                                yHigh = self.dimension - ((self.yPos + self.dimension)-frame.shape[0])

                                            #transfer the visible pixel values of the selection
                                            #to the chosenFace zeros array
                                            for i in range(yLow, yHigh):
                                                for j in range(xLow, xHigh):
                                                    for k in range(3):
                                                        self.chosenFace[i, j, k] = frame[(self.yPos + i), (self.xPos + j), k]

                                            #cv2.imwrite('test.jpg', self.chosenFace)
                                            self.endTransition = True


                                        #the selection is entirely within the frame of capture
                                        else:
                                            self.chosenFace = frame[self.yPos:self.yPos+self.dimension, self.xPos:self.xPos+self.dimension]
                                            self.chosenFace = np.array(self.chosenFace)
                                            self.endTransition = True
                                    

                #if no face detected and timer is running. stop/reset the timer
                else:
                    if self.timing:
                        self.timing = False
                 

            ##end of 'if' conditionals 
            #place the frame into the centre of the canvas for fullscreen display
            frame = cv2.resize(frame, (170, 128), interpolation = cv2.INTER_AREA)
            outputFrame[476:-476, 875:-875] = frame
            cv2.imshow(self.windowName, outputFrame)

            k = cv2.waitKey(33)

            if k == ord('q'):
                break

        #end of the while loop
        self.cap.release()
        return False





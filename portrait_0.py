#portrait_0 - a reworking of the Predictor class used for portrait_1
#and the subsequent display  Christopher Shepherd 2017

#import dependencies
import cv2
import numpy as np
import tensorflow as tf
import sys
import multiprocessing

import multiVAE
import utils

import os
#trying to manage tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

#load starting image
meanFace = cv2.imread('images/mean.png')
#meanFace = cv2.imread('0.jpg')
meanFace = np.array(meanFace)

#set up window context
cv2.namedWindow('portrait_0', cv2.WINDOW_NORMAL)
#set window to Fullscreen
cv2.setWindowProperty('portrait_0', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
windowName = 'portrait_0'

cap = cv2.VideoCapture(0)

#resize the mean face
face = cv2.resize(meanFace, (480, 480), interpolation = cv2.INTER_CUBIC)
smallerFace = cv2.resize(meanFace, (128, 128), interpolation = cv2.INTER_AREA)



#generate() loads the saved sessions and produces concurrent frames, passing them to display()
#via a queue
def generate(startFace, queue):

    #frame to be inferred from
    startFrame = startFace/255.0
    #add batch dimension
    currentFrame = startFrame.reshape([1,startFrame.shape[0], startFrame.shape[1], startFrame.shape[2]])
    #to prolong period until complete deterioration, the left-facing profile is produced directly 
    #from the right facing profile. For this to be possible it is necessary to store the 
    #right-profile image until it can be passed to the graph producing the left-profile
    holdingFrame = startFrame.reshape([1,startFrame.shape[0], startFrame.shape[1], startFrame.shape[2]])


    #for every graph stored, %47 to allow for multiple rotations
    for frame_i in range(93):

        graph_i = frame_i % 47

        #placeholder graph shape
        ae = multiVAE.VAE(input_shape=[None, 128, 128, 3],
                n_filters=[100, 100, 100],
                filter_sizes=[3, 3, 3],
                encoderNum = 0,
                n_hidden=250,
                n_code=100,
                activation=tf.nn.tanh)

        #set up and initialise the session
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        #restore the weight values of the relevant graph/frame
        sessName = 'graphData/'+str(graph_i)+'/portrait-frame'+str(graph_i)
        saver.restore(sess, sessName)

        #pass in the looking right face profile output if moving to left-profile
        if graph_i == 35:
            Xs = holdingFrame
            Ys = holdingFrame

        else:
            #input is the current frame
            Xs = currentFrame
            Ys = currentFrame

        #'x_output_final' is the result
        output = sess.run(ae['x_output_final'], feed_dict={ae['x']:Xs,
            ae['y']:Ys,
            ae['train']:False,
            ae['keep_prob']:1.0})

        #reshape to the original image shape
        outputImage = output.reshape([-1, 128, 128, 3])
        outputImage = np.array(outputImage)

        #set this output to be the next input
        currentFrame = outputImage

        #output of graph 10 is the right-profile image
        if graph_i == 10:
            holdingFrame = outputImage

        #finished frame is without the batch dimension
        finishedFrame = outputImage[0]

        #acquire Lock and append finishedFrame to the sharedImageList to be displayed
        ###-->acquire Lock
        #No longer using lock and shared memory
        #instead a queue is used to pass results between processes
        queue.put(finishedFrame)

        tf.reset_default_graph()
        sess.close()

    #queue.put(None)
    queue.close()


#display the generated frames
def display(capture, windowName, face, queue):

    #store the frames generated by the generator
    createdFrames = []
    #append the initial image passed by the Detector
    createdFrames.append(face)

    #is displayer running
    running = True

    #is the frame count increasing
    increasing = True

    #to monitor which frame is currently being shown
    count = 0
    stopperNumber = 1

    #create the canvas for the final output of portrait_0 
    portrait_0 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    while running:

        #if there is a new frame on the queue, get it and store it in the frames list
        if not queue.empty():

            newFrame = queue.get()
            newFrame = np.array(newFrame)
            newFrame = newFrame * 255.0

            createdFrames.append(newFrame)
            print('newFrame recieved')


        #place the current frame into the centre of the video capture
        ret, frame = capture.read()
        frame = cv2.resize(frame, (170, 128), interpolation = cv2.INTER_AREA)

        frame[:, 21:-21] = createdFrames[count]

        portrait_0[476:-476, 875:-875] = frame

        cv2.imshow(windowName, portrait_0) 


        ##############################7
        ##--have a max buffer value 
        ##--on increasing = False maxBuffer += 1 if maxBuffer < len(createdFrames)

        #if no frames yet generated, only display the first frame
        if len(createdFrames) == 1:
            count = 0
        #if all frames have been generated return final frame
        elif len(createdFrames) == 94 and count == 93:
            return createdFrames[count]
        #else move back and forth across the generated frames
        elif increasing and (count + 1) < stopperNumber:#len(createdFrames):
            count = count + 1
        elif increasing and (count+1) == stopperNumber:#len(createdFrames):
            increasing = False
            count = count - 1
            if stopperNumber < len(createdFrames):
                stopperNumber = stopperNumber + 1
        elif not increasing and count > 0:
            count = count -1
        else:
            increasing = True
            count = count + 1

        k = cv2.waitKey(66)

        if k == ord('q'):
            break

    return False


####----The main program----#########################################

#queue to pass generated frames to the display process
queue = multiprocessing.Queue() 

#generator process, pass the smaller initial frame to start
gen = multiprocessing.Process(target=generate, args=(smallerFace, queue))

#generator process is a daemon so that it will not interrupt if the display
#process needs to quit early(specifically used for testing as display should not
#ordinarily finish before the generator)
gen.daemon = True
gen.start()

#display function returns the final image once it has finished
finalImage = display(cap, windowName, smallerFace, queue)

#no need to join the generator process as it is a daemon
#gen.join()
#dis.join()
cap.release()
cv2.destroyAllWindows()




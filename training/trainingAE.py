#####----Training and saving graphs to map frame_i -> frame_i+1

#import necessary libraries
import os
import re
import numpy as np
import cv2
import tensorflow as tf

import utils
import multiVAE

#training videos. Which videos can be used to train which aspect of the model:
#videos 1 - 12 all positions
#video 9 = 11 - 23
#video 13 = position 0 - 24
#videos 14 - 52 = position 0-11, 35-47
#videos 24, 30, 37, 42, 51, 52. only positions 0 - 11

#list to hold training videos, a list is necessary as the raw video frames are different sizes
vidMatrix = []
#which videos can not be used for training this set of graphs
unUsed = [9, 13, 24, 30, 37, 42]
#range(1, 53)

#for each video
for vid_i in range(1, 51):

    #filter out unwanted training videos for this portion of training
    if vid_i in unUsed:
        pass

    else:
    #import the video files and make into tensors

        name = 'frames'+str(vid_i)
        vid1 = [os.path.join(name, ims) for ims in sorted(os.listdir(name))]

        #sort the video frame names into sequential order
        vid1.sort(key=lambda var:[int(x) if x.isdigit()else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        #load the images and convert to np array
        vidFrames = [cv2.imread(im_i) for im_i in vid1]
        vidFrames = np.array(vidFrames)

        #append to the training-data matrix
        vidMatrix.append(vidFrames)

#preprocess the videos
vidMatrix_Processed = []
for vid_i in range(len(vidMatrix)):
    video = []
    for im_i in range(48):
        theImage = vidMatrix[vid_i][im_i]
        smaller = cv2.resize(theImage, (128,128))
        video.append(smaller)
        
    vidMatrix_Processed.append(video)
 
vidMatrix_Processed = np.array(vidMatrix_Processed)

#load start frame for monitoring graph output at this stage of training
meanStart = cv2.imread('mean41.jpg')
mean_Start = np.array(meanStart)
meanSmaller =cv2.resize(mean_Start, (128,128)) 
meanFrame = meanSmaller/255.0

#add a batch dimension so that it can be passed to the graph
faceBatch = meanFrame.reshape([1, meanFrame.shape[0], meanFrame.shape[1], meanFrame.shape[2]])

#list to hold monitor images
holdingFrames = []
holdingFrames.append(faceBatch)

#the current range of graphs to train
for currentNumber in range(42, 47):
    frame_i = currentNumber
    #reset the graph variables and change the ckpt/saver name
    ae =  multiVAE.VAE(input_shape=[None, 128, 128, 3],
                          n_filters=[100, 100, 100],
                           filter_sizes=[3, 3, 3],
                           encoderNum = 0,
                           n_hidden=250,
                           n_code=100,
                           activation=tf.nn.tanh)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(ae['cost'])
    
    ckpt_name='graphData/' + str(currentNumber) + '/portrait-frame' + str(currentNumber)
    
    #create new session and saver
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    
    t_i = 0
    #number of batches 3 batches of 17 first quarter
    #44 videos for final quarter
    num_batches = 4 
    batch_size = 11 #numVideos per training batch
    epoch_i = 0
    n_epochs = 12501 #9901 first quarter(51 videos)
    keep_prob = 0.8
    
    #create batch_xs and corresponding y_s
    batch_xs_all = vidMatrix_Processed[:, frame_i]
    
    #if the last frame loop back around to frame 0
    if currentNumber == vidMatrix_Processed.shape[1]-1:
        batch_ys_all = vidMatrix_Processed[:, 0]
    else:
        batch_ys_all = vidMatrix_Processed[:, frame_i+1]
    
    #make values 0 - 1 rather than 0 - 255
    batch_xs_all = batch_xs_all/255.0
    batch_ys_all = batch_ys_all/255.0
    
    #for each epoch
    for epoch_i in range(n_epochs):
        
        train_i = 0
        train_cost = 0
        
        np.random.seed(epoch_i)
        
        #for each batch
        for batch_i in range(num_batches):

            #create and randomize batch order
            indices = np.random.permutation(batch_size)
            indices = indices + (batch_i * batch_size)

            batch_xs = batch_xs_all[indices]
            batch_ys = batch_ys_all[indices]
            
            #pass the batche x and targets to the graph to train(minimizing cost)
            train_cost += sess.run([ae['cost'], optimizer], feed_dict={
                ae['x']:batch_xs, ae['y']:batch_ys, ae['train']: True,
                ae['keep_prob']: keep_prob})[0]
            
        
        if epoch_i > 0 and epoch_i % 500 == 0: #900 with 51videos
            
            #to show that progress is being made print the current epoch and cost
            print('epoch:', epoch_i)
            print('average cost:', train_cost/(batch_size*2))
            
            
            ##mean Test Output, pass the testing frame to the graph and save the output
            Xs = holdingFrames[currentNumber-42]
            Ys = holdingFrames[currentNumber-42]
            
                    
            results = sess.run(ae['x_output_final'], feed_dict={ae['x']:Xs,
                                                                ae['y']:Ys,
                                                                ae['train']:False,
                                                                ae['keep_prob']: 1.0})

            results = results.reshape([-1, 128, 128, 3])
            results = np.array(results)
            
            #once training of this graph is complete(been through optimal epochs), save the session
            if epoch_i == n_epochs - 1:
                holdingFrames.append(results)
                saver.save(sess, "./"+ckpt_name,
                        global_step=epoch_i,
                        write_meta_graph=False)
                
            name = 'portrait_0/frame' + str(currentNumber) + 'epoch' + str(epoch_i) + '.jpg'

            cv2.imwrite(name, results[0]*255.0)
                    
                
    #close processing for this frame
    tf.reset_default_graph()
    sess.close()
        
    #move on the the next frame

###################################################################################################
###################################################################################################
#once all of the graphs have been trained for this session, exit
print('......Finished Training')


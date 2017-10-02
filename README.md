# imageInference

### intro:
This project demonstrates my first attempt to use neural networks (specifically a convolutional autoencoder) for the generation of art. The main aim of the project, 'portrait_0', was to infer from a starting forward-facing 'mean-face' how that face would look if it were to begin turning on the spot. The 'mean-face' is the mean-average taken from a faces dataset. 'portrait_1' provides a framework for detecting and infering from any face. However, owing to the limitations of my implementation, it is not currently successful. Ways i intend to improve this project are discussed below.

* * *
### build:
The main project dependencies are NumPy, Tensorflow and OpenCV

graphData stores the saved tf sessions - they are not included in the repo owing to their size.
In this project one autoencoder was used per-frame. This is obviously not ideal and in future i would like to adapt the project to use a recursive network.   

'portrait_1.py' makes use of 'detector.py'(detecting and cropping valid faces) and 'predictor.py'(passing images through the trained networks to produce anticipated, successive frames).
'portrait_0.py' is adapted from these.    

'multiVAE.py' defines the structure of the autoEncoder    

'training/trainingAE.py' shows how the networks were trained. As a dataset, a varying number of videos(depending on the degree of rotation) of heads turning in the centre of frame were used.    
'images' contains the initial input for portrait_1 and portrait_0, along with the example outputs shown below.    
also not included in this repo are the opencv files 'haarcascade_frontalface_alt.xml' and 'haarcascade_eye.xml' which are used for the face detection element of 'portrait_1'.    

* * *
### results as they stand:

portrait_0 outputs:
![mean1](/images/mean1.jpg) ![mean4](/images/mean4.jpg) ![mean6](/images/mean6.jpg) ![mean8](/images/mean8.jpg)    

Although some of the outputs for 'portrait_0' were quite nice(above), this project is still a work in progress.  The current methods used for training the graph are not flexible enough to allow for successful inference from live captured faces.

Future steps to improve this project include increasing the size of the dataset and, with further study, changing the model from a standard autoencoder to a generative adversarial network and making the network recurrent(using an LSTM layer).  There is also the possibility to experiment with a semi-supervised model perhaps providing markers such as approximate degree of rotation to guide the network.


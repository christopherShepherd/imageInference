###convolutional variational auto-encoder implementation adapted from Parag K. Mital
###https://github.com/pkmital/CADL/blob/master/session-3/libs/vae.py

#import dependencies
import tensorflow as tf
import numpy as np
import os
import utils
import batch_norm

def VAE(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        encoderNum = 0,
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh):

   ''' 
    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].

    n_filters : list, optional
        Number of filters for each layer.
        This refers to the total number of output filters to create for each layer, with each layer's number of output filters as a list.

    filter_sizes : list, optional
        This refers to the ksize (height and width) of each convolutional layer.

     n_hidden : int, optional
         variational.  This refers to the first fully connected layer prior to the variational embedding, directly after the encoding.  After the variational embedding, another fully connected layer is created with the same size prior to decoding.

     n_code : int, optional
        variational.  This refers to the number of latent Gaussians to sample for creating the inner most encoding.

     activation : function, optional
         Activation function to apply to each layer, e.g. tf.nn.relu

     dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.

    Returns
    -------
    model:dict
    {
        'cost': Tensor to optimize.
        'Ws': All weights of the encoder.
        'x': Input Placeholder
        'z': Inner most encoding Tensor(latent features)
        'y': target image placeholder
        'keep_prob': Amount to keep when using Dropout
        'train': Set to True when training/Applies to Batch Normalization
    }
   ''' 

   #network input placeholders
   x = tf.placeholder(tf.float32, input_shape, 'x'+str(encoderNum))
   y = tf.placeholder(tf.float32, input_shape, 'y'+str(encoderNum))

   phase_train = tf.placeholder(tf.bool, name='phase_train'+str(encoderNum))
   keep_prob = tf.placeholder(tf.float32, name='keep_prob'+str(encoderNum))
   
 
   x_tensor = x
   current_input = x_tensor

   #lists to hold the weights and shapes of each layer of the encoder
   Ws = []
   shapes = []

   #Build the encoder
   for layer_i, n_output in enumerate(n_filters):
       with tf.variable_scope(str(layer_i)+str(encoderNum)):
           shapes.append(current_input.get_shape().as_list())

           #produce weights and values through convolution
           h, W = utils.conv2d(x=current_input,
                   n_output=n_output,
                   k_h=filter_sizes[layer_i],
                   k_w=filter_sizes[layer_i],
                              name='conv2d'+str(layer_i)+str(encoderNum))
          
           #pass normalised batch through the activation function
           h = activation(batch_norm.Batch_norm(h, phase_train, 'bn'+str(layer_i)+str(encoderNum)))

           #for dropout
           h = tf.nn.dropout(h, keep_prob)

           #add the weights to the weights list
           Ws.append(W)
           #input for next layer is output for this layer
           current_input = h

   shapes.append(current_input.get_shape().as_list())


   #variational section
   with tf.variable_scope('variational'+str(encoderNum)):

       dims = current_input.get_shape().as_list()

       if len(dims) == 4:
           flattened = tf.reshape(current_input, shape=[-1, dims[1]*dims[2]*dims[3]])
       elif len(dims) == 2 or len(dims) ==1:
           flattened = current_input

       #linear fully connected layer at the centre of the encoder
       h = utils.linear(flattened, n_hidden, name='W_fc')[0]
       h = activation(batch_norm.Batch_norm(h, phase_train, 'fc/bn'))
       h = tf.nn.dropout(h, keep_prob)

       z_mu = utils.linear(h, n_code, name='mu')[0]
       z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]

       #Sample from noise distribution
       epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_code]))

       #Sample from posterior
       z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

       h = utils.linear(z, n_hidden, name='fc_t')[0]
       h = activation(batch_norm.Batch_norm(h, phase_train, 'fc_t/bn'))
       h = tf.nn.dropout(h, keep_prob)

       size = dims[1]*dims[2]*dims[3]
       h = utils.linear(h, size, name='fc_t2')[0]
       current_input = activation(batch_norm.Batch_norm(h, phase_train, 'fc_t2/bn'))

       current_input = tf.reshape(current_input, tf.stack([tf.shape(current_input)[0],
           dims[1], dims[2], dims[3]]))

   #reverse the shapes filters and weights to undo the encoding
   shapes.reverse()
   n_filters.reverse()
   Ws.reverse()

   n_filters += [input_shape[-1]]



   ###Decoding-------------------
   for layer_i, n_output in enumerate(n_filters[1:]):
       with tf.variable_scope('decoder/{}'.format(layer_i)+str(encoderNum)):
           shape = shapes[layer_i +1]

           #convolve
           h, W = utils.deconv2d(x=current_input,
                   n_output_h=shape[1],
                   n_output_w=shape[2],
                   n_output_ch=shape[3],
                   n_input_ch=shapes[layer_i][3],
                   k_h=filter_sizes[layer_i],
                   k_w=filter_sizes[layer_i])
          
           h = activation(batch_norm.Batch_norm(h, phase_train, 'dec/bn'+str(layer_i)))

           #for dropout
           h = tf.nn.dropout(h, keep_prob)

           current_input = h

   #the output from the final decoding layer is the output of the graph
   x_output = current_input

   #flatten the target image
   y_flat = utils.flatten(y)

##make the model learn an output which when added to the original input
##produces the next frame

   #flatten the input image
   x_original_flat = utils.flatten(x)

   #flatten the graph output
   dims1 = x_output.get_shape().as_list()
   if len(dims1) == 4:
       x_output_flat = tf.reshape(x_output,shape=[-1, dims1[1] * dims1[2] * dims1[3]])
   elif len(dims1) == 2 or len(dims1) == 1:
       x_output_flat = x_output
        
   
   #the ultimate output is the graph output added to the original input
   x_output_final = x_original_flat + x_output_flat  

   #l2 loss
   #difference between final output and target image
   loss_x = tf.reduce_sum(tf.squared_difference(y_flat, x_output_final), 1)

   #penalizing latent vectors
   loss_z = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_log_sigma - 
           tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)

   #total cost is the of the image loss and the latent loss
   cost = tf.reduce_mean(loss_x + loss_z)

   return {'cost': cost, 'Ws': Ws, 'x':x, 'x_output_final':x_output_final, 'z':z, 'y':y,
            'keep_prob':keep_prob, 'train':phase_train}


















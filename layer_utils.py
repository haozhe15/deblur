import numpy as np 
import tensorflow as tf 

def res_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    """
    Instanciate a Resnet Block using tf.layers.

    :param input: Input tensor
    :param filters: Number of output filters in the convolution
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Output tensor
    """
    paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
    # Use tf.pad directly instead of using ReflectionPadding2D
    x_1 = tf.pad(inputs, paddings, "REFLECT")
    conv_1 = tf.layers.conv2d(inputs=x_1,filters=filters,kernel_size=kernel_size,strides=strides)
    norm_1 = tf.contrib.layers.instance_norm(conv_1)
    act = tf.nn.relu(norm_1)

    if use_dropout:
        act = tf.layers.dropout(act,rate=0.5)

    x_2 = tf.pad(act, paddings, "REFLECT")
    conv_2 = tf.layers.conv2d(inputs=x_2,filters=filters,kernel_size=kernel_size,strides=strides)
    norm_2 = tf.contrib.layers.instance_norm(conv_2)

    outputs = tf.add(norm_2, inputs)  

    return outputs

if __name__ == "__main__":
    # Simple test
    tf.reset_default_graph()
    inputs=tf.constant([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]], dtype=tf.float32) 
    inputs=tf.reshape(inputs,[1,4,4,1])
    outputs=res_block(inputs,20)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(inputs).shape)
        print(sess.run(outputs).shape)
      
        

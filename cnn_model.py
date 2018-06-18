import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, flatten, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def conv_maxpool(x, num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    layer = conv_2d(x, num_outputs, conv_ksize, conv_strides, padding='SAME')
    layer = max_pool_2d(x, pool_ksize, pool_strides, padding='SAME')
    return layer


def build_model(x, keep_prob):
    conv_ksize = (3, 3)
    conv_strides = (2, 2)
    pool_ksize = (2, 2)
    pool_strides = (2, 2)

    # First convolution & maxpooling
    layer = conv_maxpool(x, 64, conv_ksize, conv_strides, pool_ksize, pool_strides)

    # Second convolution & maxpooling
    layer = conv_maxpool(x, 64, conv_ksize, conv_strides, pool_ksize, pool_strides)

    # Flatten Layer
    layer = flatten(x)

    # Fully connect layer
    layer_output = fully_connected(layer, 1024)
    layer_output = dropout(layer_output, keep_prob)
    layer_output = fully_connected(layer_output, 512)
    # layer_output = dropout(layer_output, keep_prob)
    layer_output = fully_connected(layer_output, 10)

    return layer_output


def run_cnn(image_shape, keep_prob, batch_size, num_class):
    new_shape = image_shape.copy()
    new_shape.insert(0, None)
    new_shape.insert(len(new_shape), 1)
    tensor_x = tf.placeholder(tf.float32, shape=new_shape, name='tensor_x')
    print(tensor_x)
    tensor_y = tf.placeholder(tf.float32, shape=[None, num_class], name='tensor_y')

    # y_pred = build_model(tensor_x, keep_prob)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=tensor_y))

    # train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)



import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, flatten, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


def conv_maxpool(x, num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    layer = conv_2d(x, num_outputs, conv_ksize, conv_strides, padding='SAME')
    layer = max_pool_2d(x, pool_ksize, pool_strides, padding='SAME')
    return layer


def build_model(x, keep_prob, conv_ksize, conv_strides, pool_ksize, pool_strides):
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


def run_cnn(
        image_shape,
        x_train,
        y_train,
        x_valid,
        y_valid,
        conv_ksize,
        conv_strides,
        pool_ksize,
        pool_strides,
        epochs,
        keep_prob,
        batch_size,
        num_class):
    new_shape = image_shape.copy()
    new_shape.insert(0, None)
    tensor_x = tf.placeholder(tf.float32, shape=new_shape, name='tensor_x')
    tensor_y = tf.placeholder(tf.float32, shape=[None, num_class], name='tensor_y')
    tensor_prob = tf.placeholder(tf.float32, name='keep_prob')

    y_pred = build_model(tensor_x, keep_prob, conv_ksize, conv_strides, pool_ksize, pool_strides)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=tensor_y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            total_cost = 0
            for i in range(0, len(x_train), batch_size):
                batch_data = x_train[i:i + batch_size, :]
                batch_onehot_vals = y_train[i:i + batch_size, :]
                _, c = s.run([optimizer, cost],
                             feed_dict={tensor_x: batch_data, tensor_y: batch_onehot_vals, tensor_prob: keep_prob})
                total_cost += c
            print('Epoch {:>2}, Distracted driver Batch {}:  '.format(epoch + 1, i), end='')
            loss = s.run(cost, feed_dict={tensor_x: x_train, tensor_y: y_train, tensor_prob: 1.0})
            correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(tensor_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            valid_accuracy = s.run(accuracy, feed_dict={
                tensor_x: x_valid,
                tensor_y: y_valid,
                tensor_prob: 1.})
            print('Cost: {:<8.3f} Valid Accuracy: {:<5.3f}'.format(
                loss,
                valid_accuracy))

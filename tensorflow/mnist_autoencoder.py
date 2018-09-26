""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

BATCH_SIZE = 50
GRID_ROWS = 5
GRID_COLS = 10
USE_RELU = True
num_steps = 100000
display_step = 1000

def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(previous, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b


def autoencoder(x, squeeze=2):
    # first fully connected layer with 50 neurons using tanh activation
    l1 = tf.nn.tanh(fc_layer(x, 28*28, 50))
    # second fully connected layer with 50 neurons using tanh activation
    l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
    # third fully connected layer with 2 neurons
    l3 = fc_layer(l2, 50, squeeze)
    # fourth fully connected layer with 50 neurons and tanh activation
    l4 = tf.nn.tanh(fc_layer(l3, squeeze, 50))
    # fifth fully connected layer with 50 neurons and tanh activation
    l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
    # readout layer
    if USE_RELU:
        out = tf.nn.relu(fc_layer(l5, 50, 28*28))
    else:
        out = fc_layer(l5, 50, 28*28)
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, l3


def main():
    # initialize the data
    mnist = input_data.read_data_sets('/tmp/MNIST_data')

    # placeholders for the images
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # build the model
    loss, output, latent = autoencoder(x,squeeze=2)

    # and we use the Adam Optimizer for training
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # We want to use Tensorboard to visualize some stuff
    #writer, summary_op = create_summaries(loss, x, latent, output)

    first_batch = mnist.train.next_batch(BATCH_SIZE)

    # Run the training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, int(num_steps+1)):
            batch, _ = mnist.train.next_batch(BATCH_SIZE)
            #feed = {x : batch[0]}

            _, l = sess.run([optimizer, loss], feed_dict={x:batch})
            #train_step.run(feed_dict=feed)
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))


        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))
        for i in range(n):
            # MNIST test set
            batch_x, _ = mnist.test.next_batch(n)
            # Encode and decode the digit image
            g = sess.run(output, feed_dict={x: batch_x})

            # Display original images
            for j in range(n):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    g[j].reshape([28, 28])

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show(block=False)

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()

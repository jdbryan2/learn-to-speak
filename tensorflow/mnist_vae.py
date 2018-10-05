from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/jacob/Projects/Data/MNIST_data")

LOAD = True
tf.reset_default_graph()


batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels / 2)


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img



sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


flavor_saver = tf.train.Saver()
if LOAD:
    flavor_saver.restore(sess, '/home/jacob/Desktop/model.ckpt')
    
else:
    for i in range(400):
        batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
            
        if not i % 200:
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
            plt.show(block=False)
            plt.imshow(d[0], cmap='gray')
            plt.show(block=False)
            print(i, ls, np.mean(i_ls), np.mean(d_ls))

    flavor_saver.save(sess, '/home/jacob/Desktop/model.ckpt')

randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
print(randoms)
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs_view = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))] 

est_randoms = sess.run(sampled, feed_dict = {X_in:imgs, keep_prob:1.0})
est_randoms2 = sess.run(mn, feed_dict = {X_in:imgs, keep_prob:1.0})
print(est_randoms)

for k in range(len(randoms)):
    plt.plot(randoms[k])
    plt.plot(est_randoms2[k])
    plt.figure()
    plt.plot(randoms[k]-est_randoms2[k])
    plt.title('Error')
    plt.show()


#for img in imgs:
#    plt.figure(figsize=(1,1))
#    plt.axis('off')
#    plt.imshow(img, cmap='gray')
#
#    plt.show(block=False)

plt.show()

# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:	2016-05-15 15:55:32
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-15 19:19:51
# @Description: Stochastic Gradient Descent

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Fixed parameters
image_size = 28**2
num_labels = 10

# Hyper parameters
learning_rate = 0.5
batch_size = 100
num_steps = 10**3

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def accuracy(predictions, labels):
	return(100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# load data
Xr = mnist.train.images
yr = mnist.train.labels

Xv = mnist.validation.images
yv = mnist.validation.labels

Xe = mnist.test.images
ye = mnist.test.labels

graph = tf.Graph()
with graph.as_default():

	tf_Xr = tf.placeholder(tf.float32, [None, image_size])
	tf_yr = tf.placeholder(tf.float32, [None, num_labels])
	
	tf_Xv = tf.constant(Xv, tf.float32)
	tf_Xe = tf.constant(Xe, tf.float32)
	
	weights = weight_variable([image_size, num_labels])
	biases = bias_variable([num_labels])
	
	logits = tf.matmul(tf_Xr, weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_yr))
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	
	train_prediction = tf.nn.softmax(tf.matmul(tf_Xr, weights) + biases)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_Xv, weights) + biases)
	test_prediction  = tf.nn.softmax(tf.matmul(tf_Xe, weights) + biases)

with tf.Session(graph=graph) as session:
	
	tf.initialize_all_variables().run()
	print('Initialized')
	
	for step in range(num_steps):
		batch = mnist.train.next_batch(batch_size)
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict={tf_Xr: batch[0], tf_yr: batch[1]})

		if (step % 100 == 0):
			print('Loss at step %d: %f' % (step, l))
			print('Training accuracy: %.1f%%' % accuracy(predictions, batch[1]))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), yv), end="\n\n")
	
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), ye))


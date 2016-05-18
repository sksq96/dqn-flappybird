# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:	2016-05-15 15:55:32
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-17 20:50:26
# @Description: Stochastic Gradient Descent

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Network parameters
n_input = 28**2
num_channels = 1 # grayscale
n_hidden_1 = 32
n_hidden_2 = 32
n_hidden_3 = 64
n_output = 10
dropout = 0.75

# Hyper parameters
learning_rate = 0.001
batch_size = 128
num_steps = 10**3

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, b):
	return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def matmul(x, W, b):
	return tf.nn.relu(tf.matmul(x, W) + b)

def accuracy(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# return(100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# Create model
def multilayer_perceptron(x, weights, biases, dropout):

	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layer and max-pooling
	conv_1 = conv2d(x, weights['l1'], biases['l1'])
	conv_1 = max_pool_2x2(conv_1)
	# conv_1 = tf.nn.dropout(conv_1, dropout)

	# Convolution Layer and max-pooling
	conv_2 = conv2d(conv_1, weights['l2'], biases['l2'])
	conv_2 = max_pool_2x2(conv_2)
	# conv_2 = tf.nn.dropout(conv_2, dropout)

	# Fully connected layer
	fcl_1 = tf.reshape(conv_2, [-1 ,7*7*n_hidden_2])
	fcl_1 = matmul(fcl_1, weights['l3'], biases['l3'])
	# fcl_1 = tf.nn.dropout(fcl_1, dropout)

	# Output, class prediction
	# bad-accuracy with RELU @ last layer
	# predictions = matmul(fcl_1, weights['out'], biases['out'])
	predictions = tf.matmul(fcl_1, weights['out'])+ biases['out']
	return predictions



# form tf-graph
graph = tf.Graph()
with graph.as_default():

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_output])
	dropout_prob = tf.placeholder(tf.float32)
		
	weights = {
		'l1': weight_variable([5, 5, num_channels, n_hidden_1]),
		'l2': weight_variable([5, 5, n_hidden_1, n_hidden_2]),
		'l3': weight_variable([7*7*n_hidden_2, n_hidden_3]),
		'out': weight_variable([n_hidden_3, n_output])
	}
	
	biases = {
		'l1': bias_variable([n_hidden_1]),
		'l2': bias_variable([n_hidden_2]),
		'l3': bias_variable([n_hidden_3]),
		'out': bias_variable([n_output])
	}
	
	# final layer
	prediction = multilayer_perceptron(x, weights, biases, dropout_prob)
	
	# starter_learning_rate = 1.
	# global_step = tf.Variable(0, trainable=False)
	# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
	
	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) + tf.nn.l2_loss(prediction)
	# loss = tf.reduce_sum(tf.nn.l2_loss(prediction-y)) # L2 loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	
	# test_prediction  = tf.nn.softmax(tf.matmul(tf_Xe, weights) + biases)

# start tf-session
with tf.Session(graph=graph) as session:
	
	tf.initialize_all_variables().run()
	print('Initialized')
	
	for step in range(num_steps):
		batch = mnist.train.next_batch(batch_size)
		_, l = session.run([optimizer, loss], feed_dict={x: batch[0], y: batch[1], dropout_prob:dropout})
		
		# if (step % 100 == 0):
			# print('Loss at step %d: %f' % (step, l))
			# print('Validation accuracy: %.1f%%' % accuracy(predictions, ), end="\n\n")
	
	print("Optimization Finished!")	
	print("Accuracy:", accuracy(prediction, y).eval({x: mnist.test.images, y: mnist.test.labels, dropout_prob:1.}))




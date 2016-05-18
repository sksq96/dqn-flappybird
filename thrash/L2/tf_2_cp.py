# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:	2016-05-15 15:55:32
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-15 19:13:16
# @Description: Stochastic Gradient Descent

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Network parameters
n_input = 28**2
n_hidden_1 = 1024
n_output = 10

# Hyper parameters
learning_rate = 0.5
batch_size = 100
num_steps = 500

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def accuracy(predictions):
	
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
	return tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	# return(100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def multilayer_perceptron(x, weights, biases):
	layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
	predictions = tf.matmul(layer_1, weights['out']) + biases['out']
	return predictions

# form tf-graph
graph = tf.Graph()
with graph.as_default():

	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_output])
		
	weights = {
		'h1': weight_variable([n_input, n_hidden_1]),
		'out': weight_variable([n_hidden_1, n_output])
	}
	
	biases = {
		'b1': bias_variable([n_hidden_1]),
		'out': bias_variable([n_output])
	}
	
	prediction = multilayer_perceptron(x, weights, biases)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	
	# test_prediction  = tf.nn.softmax(tf.matmul(tf_Xe, weights) + biases)

# start tf-session
with tf.Session(graph=graph) as session:
	
	tf.initialize_all_variables().run()
	print('Initialized')
	
	for step in range(num_steps):
		batch = mnist.train.next_batch(batch_size)
		_, l = session.run([optimizer, loss], feed_dict={x: batch[0], y: batch[1]})

		if (step % 100 == 0):
			print('Loss at step %d: %f' % (step, l))
			print("Training Accuracy:", accuracy(prediction).eval({x: batch[0], y: batch[1]}))
			print("Validation Accuracy:", accuracy(prediction).eval({x: mnist.validation.images, y: mnist.validation.labels}), end="\n\n")
	
	print("Optimization Finished!")	
	print("Accuracy:", accuracy(prediction).eval({x: mnist.test.images, y: mnist.test.labels}))



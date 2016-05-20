# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2016-05-20 11:27:32
# @Last Modified by:   shubham
# @Last Modified time: 2016-05-20 18:55:38

import tensorflow as tf

# Global parameters
IMAGE_SIZE = 80
ACTIONS = 2

# Network parameters
STRIDE_1 = 4
STRIDE_2 = 2
STRIDE_3 = 1
HIDDEN_1 = 32
HIDDEN_2 = 64
HIDDEN_3 = 64
HIDDEN_4 = 512
PATCH_SIZE_1 = 8
PATCH_SIZE_2 = 4
PATCH_SIZE_3 = 3

# Hyper parameters
HISTORY_LENGTH = 4
LEARNING_RATE = 1e-6

# Deep Neural Network helper functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, b, stride):
	return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def matmul(x, W, b):
	return tf.nn.relu(tf.matmul(x, W) + b)

def accuracy(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# return(100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# Create network
def create_network():
	
	weights = {
		'c1': weight_variable([PATCH_SIZE_1, PATCH_SIZE_1, HISTORY_LENGTH, HIDDEN_1]),
		'c2': weight_variable([PATCH_SIZE_2, PATCH_SIZE_2, HIDDEN_1,  HIDDEN_2]),
		'c3': weight_variable([PATCH_SIZE_3, PATCH_SIZE_3, HIDDEN_2,  HIDDEN_3]),
		'fc1': weight_variable([(5*5)*HIDDEN_3, HIDDEN_4]),
		'fc2': weight_variable([HIDDEN_4, ACTIONS])
	}
	
	biases = {
		'c1': bias_variable([HIDDEN_1]),
		'c2': bias_variable([HIDDEN_2]),
		'c3': bias_variable([HIDDEN_3]),
		'fc1': bias_variable([HIDDEN_4]),
		'fc2': bias_variable([ACTIONS])
	}

	# Input image
	x = tf.placeholder("float", shape=[None, IMAGE_SIZE, IMAGE_SIZE, HISTORY_LENGTH])
	
	# Convolution Layers and max-pooling
	conv_1 = conv2d(x, weights['c1'], biases['c1'], STRIDE_1)
	conv_1 = max_pool_2x2(conv_1)
	# conv_1 = tf.nn.dropout(conv_1, dropout)

	conv_2 = conv2d(conv_1, weights['c2'], biases['c2'], STRIDE_2)
	# conv_2 = max_pool_2x2(conv_2)
	# conv_2 = tf.nn.dropout(conv_2, dropout)

	conv_3 = conv2d(conv_2, weights['c3'], biases['c3'], STRIDE_3)
	# conv_3 = max_pool_2x2(conv_3)
	# conv_3 = tf.nn.dropout(conv_3, dropout)

	# Fully connected layer
	fcl_1 = tf.reshape(conv_3, [-1 ,(5*5)*HIDDEN_3])
	fcl_1 = matmul(fcl_1, weights['fc1'], biases['fc1'])
	# fcl_1 = tf.nn.dropout(fcl_1, dropout)

	# Output, actions prediction
	actions = tf.matmul(fcl_1, weights['fc2'])+ biases['fc2']
	return actions


def optimize_network(actions):
	
	# form tf-graph
	graph = tf.Graph()
	
	with graph.as_default():
		# tf Graph input
		x = tf.placeholder(tf.float32, [None, n_input])
		y = tf.placeholder(tf.float32, [None, n_output])
		dropout_prob = tf.placeholder(tf.float32)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(actions, y))
		optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


# def 


def main():
	actions = create_network()
	optimize_network(actions)

if __name__ == '__main__':
	main()


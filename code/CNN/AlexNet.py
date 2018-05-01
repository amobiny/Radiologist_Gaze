"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: AlexNet implemented on the MNIST data.

The general structure of the network is similar to the network used in the original
paper: "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et. al, 
of course with changes in the network parameters such as the number of convolutional layer filters, 
kernel sizes, etc. to make it compatible with the MNIST data.
To use it with your own data, feel free to change the parameters such as kernel sizes, strides, etc.
**********************************************************************************
"""

import tensorflow as tf
from ops import *


class Alexnet:
    # Class properties
    __network = None         # Graph for AlexNet
    __train_op = None        # Operation used to optimize loss function
    __loss = None            # Loss function to be optimized, which is based on predictions
    __accuracy = None        # Classification accuracy
    __probs = None           # Prediction probability matrix of shape [batch_size, numClasses]

    def __init__(self, numClass, imgSize, imgChannel):
        self.imgSize = imgSize
        self.numClass = numClass
        self.imgChannel = imgChannel
        self.h1 = 200		# Number of neurons in the first fully-connected layer
        self.h2 = 75		# Number of neurons in the second fully-connected layer
        self.init_lr = 0.001	# Initial learning rate
        self.x, self.y, self.keep_prob = self.create_placeholders()
        self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.imgSize, self.imgSize, self.imgChannel),
                                    name='x-input')
            self.y = tf.placeholder(tf.float32, shape=(None, self.numClass), name='y-input')
            self.keep_prob = tf.placeholder(tf.float32)
        return self.x, self.y, self.keep_prob

    def inference(self):
        if self.__network:
            return self
        # Building network...
        with tf.variable_scope('AlexNet'):
            net = conv_2d(self.x, 4, 1, 16, 'CONV1',
                          is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = max_pool(net, 2, 2, 'MaxPool1')
            net = conv_2d(net, 3, 1, 32, 'CONV2',
                          is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = max_pool(net, 2, 2, 'MaxPool2')
            net = conv_2d(net, 3, 1, 64, 'CONV3',
                          is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = conv_2d(net, 3, 1, 64, 'CONV4',
                          is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = conv_2d(net, 3, 1, 32, 'CONV5',
                          is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = max_pool(net, 2, 2, 'MaxPool3')
            layer_flat = flatten_layer(net)
            net = fc_layer(layer_flat, self.h1, 'FC1',
                           is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = dropout(net, self.keep_prob)
            net = fc_layer(net, self.h2, 'FC2',
                           is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=True)
            net = dropout(net, self.keep_prob)
            net = fc_layer(net, self.numClass, 'FC3',
                           is_train=self.is_train, batch_norm=True, add_reg=False, use_relu=False)
            self.__network = net
        return self

    def pred_func(self):
        if self.__probs:
            return self
        self.__probs = tf.nn.softmax(self.__network)
        return self

    def accuracy_func(self):
        if self.__accuracy:
            return self
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.__network, 1), tf.argmax(self.y, 1))
            self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.__accuracy)
        return self

    def loss_func(self):
        if self.__loss:
            return self
        with tf.name_scope('Loss'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.__network)
            self.__loss = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', self.__loss)
        return self

    def train_func(self):
        if self.__train_op:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    @property
    def network(self):
        return self.__network

    @property
    def probs(self):
        return self.__probs

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:04:11 2019

# Attention Model - decoder is the same as the machine translation
# Steps:
#   1. extract the features from the lower convolutional layer of 
#       InceptionV3 giving us a vector of shape (8, 8, 2048).
#   2. You squash that to a shape of (64, 2048).
#   3. vector is then passed through the CNN Encoder 
#       (which consists of a single Fully connected layer).
#   4. RNN (GRU) attends over the image to predict the next word.

@author: alejandrogonzales
"""

import tensorflow as tf
import numpy as np

class NeuralAttention(tf.keras.Model):
    def __init__(self, units):
        super(NeuralAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        
        # hidden shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
    
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights
        
        
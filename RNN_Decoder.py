#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:04:13 2019

RNN Encoder using GRU for taking CNN encoded images and decoding
the captions for the particular image

@author: alejandrogonzales
"""

import tensorflow as tf
from NeuralAttention import NeuralAttention

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, batch_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        
        # input layer: trainable lookup table that maps the numbers of
        # each word to a vector with embedding_dim dimensions
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   mask_zero=True,
                                                   batch_input_shape=[
                                                           batch_size, None])
#                                                   trainable=True)
    
        # GRU can have multiple models for better decoding/prediction of text
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        
        self.attention = NeuralAttention(self.units)
        
    def call(self, x, features, hidden):
        # defining attention as separate neural model
        context_vector, attention_weights = self.attention(features, hidden)
        
        # x shape after passing through embedding == 
        # (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == 
        # (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # pass the concatenated vector to the GRU with the procedure:
        # 1. For each word the model looks up embedding
        # 2. Runs GRU one timestep with embedding as input
        # 3. Applies dense layer to generate logits predicting
        #   the log-likelihood of the next word
        
        output, state = self.gru(x)
        
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        
        return x, state, attention_weights
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
                
        
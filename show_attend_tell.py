from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm 

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

# InceptionV3 weights for model creation
inceptionv3_weights = '/Users/alejandrogonzales/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Training and annotation paths
trains = '../cocodataset/train2017'
anns = '../cocodataset/annotations'
train_path = os.path.abspath(trains)
ann_path = os.path.abspath(anns)
ann_file = ann_path + '/captions_train2017.json'

print("train_path = ", train_path)
print("ann_path = ", ann_path)
print("ann_file = ", ann_file)

# Pre-process images using InceptionV3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# read the json file
with open(ann_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

print("Annotations:")
print("keys:")
print(annotations.keys())
print("\n")
#print("JSON:")

# Loop through annotations and create images and captions
count = 0
# N = len(annotations['annotations'])
N = 100
for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = train_path + '/%012d.jpg' % (image_id)
   
    # append values to the image and caption vectors
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

    # TODO Remove this when done seeing the system
    if (count == N):
        break 
    count = count + 1
print("\n")

# Shuffle captions and image names together (set random state for both)
train_captions, img_name_vector = shuffle(all_captions,
        all_img_name_vector,
        random_state=1)
print("train_captions len = ", len(train_captions))
print("\n")

# Select first 30000 from the shuffled data sets
num_samples = 30000
train_captions = train_captions[:num_samples]
img_name_vector = img_name_vector[:num_samples]
print("Train captions len = ", len(train_captions))
print("Img name len = ", len(img_name_vector))
print("\n")

# Create CNN (InceptionV3) where the output layer is the last CNN layer for 
# classification
# Shape of the output layer is 8x8x2048
# Steps:
#   1. Use the last CNN layer because we are using attention for our purpose
#   2. No initialization during training due to bottleneck
#   3. Forward each image through the network and store resulting vector in 
#       dictionary (image_name -> feature_vector)
#   4. After all images are passed through the network, pickle dictionary and 
#       save to disk (uses serialization to save dictionary data)

# Load the model directly from .h5 file
image_model = tf.keras.applications.InceptionV3(weights=inceptionv3_weights, 
                                                include_top=False)
print("Image model:")
print(image_model.summary())
print("\n")

new_input = image_model.input
hidden_layer = image_model.layers[-1].output
print("InceptionV3 Architecture:")
print("new input shape = ", new_input.shape)
print("hidden_layer shape = ", hidden_layer.shape)
print("\n")

# Create new model using input and output layers of inception
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
print("Image Extract Model:")
print(image_features_extract_model.summary())
print("\n")

# Cache features extracted from InceptionV3
# Notes:
#   1. Caching output in RAM is too much (requires 8*8*2048 floats per image)
#   2. Cache using the checkpoint callback used in Shakespeare learning
# Get unique images
BATCH_SIZE = 16
encode_train = sorted(set(img_name_vector))

# Create a tensorflow dataset (see Shakespeare)
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        BATCH_SIZE)
print("Image dataset:")
print(image_dataset)
print("\n")

# Loop through images and feed to our InceptionV3 model to extract features
for img, path in image_dataset:
    batch_features = image_features_extract_model(img)
    # print("batch_features_shape = ", batch_features.shape)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, 
                                 batch_features.shape[3]))
    
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        bf_numpy = bf.numpy()
        # print("BF numpy shape = ", bf_numpy.shape)
        np.save(path_of_feature, bf_numpy)

# Pre-process and tokenize the captions
# Steps:
#   1. Tokenize captions (splitting spaces) -> unique word vocab
#   2. Limit vocab to top 5000 words to save memory. Replace all other
#       tokens with "UNK" for unknown
#   3. Create word-to-index and index-to-word mappings
#   4. Pad all sequences to be same length as the longest one
# Find max length of any caption in the dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose top 5000 words from vocab (i.e. words per image frame)
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token='<unk>',
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
print("Training Sequences (captions):")
print(len(train_seqs))
print(len(train_seqs[0]))
print(train_seqs)
print("\n")

# word to index and vice versa
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)
print("Training Sequences (captions):")
print(len(train_seqs))
print(len(train_seqs[0]))
print(train_seqs)
print("\n")

# Pad each vector to the max length of the captions
# If max_length not provided, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
    train_seqs, padding='post')
print("Caption vector:")
print(cap_vector.shape)
print(cap_vector)
print("\n")

# Max length to store the attention weights
max_len = calc_max_length(train_seqs)
print("Max len = ", max_len)
print("\n")

# Split data into training and testing 70/30 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    img_name_vector, cap_vector, test_size=0.3, random_state=0)

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

# Create tf dataset for training (batch_size, embedding and rnn units)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE

print("Vocab size = ", vocab_size)
print("Number of steps = ", num_steps)
print("\n")

# Shape of the vector extracted from InceptionV3 is (64, 2048)
features_shape = 2048           # number of output features
attention_features_shape = 64   # dimensions of each feature

# Load the numpy files
def map_func(img_name, cap):
    print("img_name = ", img_name)
    print("caption = ", cap)
    print("\n")
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap

# Slice the images and captions into a new Dataset tensor
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
print("Dataset:")
print(dataset)
print("\n")

# Use map to load the numpy files in parallel (this merges both the image
# 2D data as well as the caption vectors into a single Dataset)
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch the training data for images and captions
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# PREFETCHING:
# Prefetching overlaps the preprocessing and model execution of a 
# training step. While the model is executing training step s, 
# the input pipeline is reading the data for step s+1. 

# Transformation uses a background thread and an internal buffer to 
# prefetch elements from the input dataset ahead of the time they are 
# requested. The number of elements to prefetch should be equal to 
# (or possibly greater than) the number of batches consumed by a 
# single training step.
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Data set iteration
for elem in dataset:
    print(elem)
    print("\n")

# Model - decoder is the same as the machine translation
# Steps:
#   1. extract the features from the lower convolutional layer of 
#       InceptionV3 giving us a vector of shape (8, 8, 2048).
#   2. You squash that to a shape of (64, 2048).
#   3. vector is then passed through the CNN Encoder 
#       (which consists of a single Fully connected layer).
#   4. RNN (GRU) attends over the image to predict the next word.





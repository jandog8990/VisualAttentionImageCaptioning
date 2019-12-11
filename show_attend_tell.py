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
for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = train_path + '/%012d.jpg' % (image_id)
   
    # append values to the image and caption vectors
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

    # TODO Remove this when done seeing the system
    print(full_coco_image_path)
    if (count == 10):
        break 
    count = count + 1
print("\n")

# Shuffle captions and image names together (set random state for both)
train_captions, img_name_vector = shuffle(all_captions,
        all_img_name_vector,
        random_state=1)

# Select first 30000 from the shuffled data sets
num_samples = 30000
train_captions = train_captions[:num_samples]
img_name_vector = img_name_vector[:num_samples]

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
    print("batch_features_shape = ", batch_features.shape)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, 
                                 batch_features.shape[3]))
    
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        bf_numpy = bf.numpy()
        print("BF numpy shape = ", bf_numpy.shape)
        np.save(path_of_feature, bf_numpy)



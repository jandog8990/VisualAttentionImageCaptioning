from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

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

# Training and annotation paths
trains = '../cocodataset/val2017'
anns = '../cocodataset/annotations'
train_path = os.path.abspath(trains)
ann_path = os.path.abspath(anns)
ann_file = ann_path + '/captions_train2017.json'

print("train_path = ", train_path)
print("ann_path = ", ann_path)
print("ann_file = ", ann_file)

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
#count = 0

# Loop through annotations and create images and captions
for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = train_path + '/%012d.jpg' % (image_id)
   
    # append values to the image and caption vectors
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

    '''
    print(full_coco_image_path)
    if (count == 10):
        break 
    count = count + 1
    '''
#print("\n")

# Shuffle captions and image names together (set random state for both)
train_captions, img_name_vector = shuffle(all_captions,
        all_img_name_vector,
        random_state=1)

# Select first 30000 from the shuffled data sets
num_samples = 30000
train_captions = train_captions[:num_samples]
img_name_vector = img_name_vector[:num_samples]

# Create CNN (InceptionV3) where the output layer is the last CNN layer for classification
# Shape of the output layer is 8x8x2048
# Steps:
#   1. Use the last CNN layer because we are using attention for our purpose
#   2. No initialization during training due to bottleneck
#   3. Forward each image through the network and store resulting vector in dictionary (image_name -> feature_vector)
#   4. After all images are passed through the network, pickle dictionary and save to disk (uses serialization to save dictionary data)
image_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
print("InceptionV3 Architecture:")
print("new input shape = ", new_input.shape)
print("hidden_layer shape = ", hidden_layer.shape)
print("\n")

# Create new model using input and output layers of inception
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
print("Image Extract Model:")
print(image_features_extract_model)
print("\n")

# Cache features extracted from InceptionV3
# Notes:
#   1. Caching output in RAM is too much (requires 8*8*2048 floats per image)
#   2. Cache using the checkpoint callback used in Shakespeare learning
checkpoint_dir = './training_checkpoints'

# names of checkpoint files (saves the model and params as the model is trained)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

# Model fit will save the weights and params from the training (however the fit is done)
#EPOCHS=10
#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])






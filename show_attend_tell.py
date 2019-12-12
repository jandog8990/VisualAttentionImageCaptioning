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

#import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
#import pickle

# Custom class files for CNN, RNN, GRU, LSTM
from CNN_Encoder import CNN_Encoder
from RNN_Decoder import RNN_Decoder

# InceptionV3 weights for model creation
inceptionv3_weights = '/Users/alejandrogonzales/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Training and annotation paths
trains = '../cocodataset/train2017'
#trains = '../cocodataset/val2017'
anns = '../cocodataset/annotations'
train_path = os.path.abspath(trains)
ann_path = os.path.abspath(anns)
ann_file = ann_path + '/captions_train2017.json'
#ann_file = ann_path + '/captions_val2017.json'

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
N = len(annotations['annotations'])
#N = 256
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

# Shuffle captions and image names together (set random state for both)
train_captions, img_name_vector = shuffle(all_captions,
        all_img_name_vector,
        random_state=1)

# Select first 30000 from the shuffled data sets
#num_samples = 30000
#num_samples = 10240
#num_samples = 4096
#num_samples = 2048
num_samples = 8
train_captions = train_captions[:num_samples]
img_name_vector = img_name_vector[:num_samples]
print("train_captions len = ", len(train_captions))
print("img_name_vector len = ", len(img_name_vector))
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
#print("Image dataset:")
#print(image_dataset)
#print("\n")

# Loop through images and feed to our InceptionV3 model to extract features
for img, path in tqdm(image_dataset):
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
#print("Training Sequences (captions):")
#print(len(train_seqs))
#print(len(train_seqs[0]))
#print(train_seqs)
#print("\n")

# word to index and vice versa
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)
print("Training Sequences (captions):")
print(len(train_seqs))
print(len(train_seqs[0]))
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

# Split data into training and testing 70/30 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    img_name_vector, cap_vector, test_size=0.3, random_state=0)

print("Train/Test Output lengths (img/cap name and values respectively):")
max_len, len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

# Create tf dataset for training (batch_size, embedding and rnn units)
#BATCH_SIZE = 64    # needed for GPU cores to pass multiple instances
BATCH_SIZE = 8
BUFFER_SIZE = 1024
embedding = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
print("Batch size = ", BATCH_SIZE)
print("Vocab size = ", vocab_size)
print("Number of steps = ", num_steps)
print("\n")

# Shape of the vector extracted from InceptionV3 is (64, 2048)
features_shape = 2048           # number of output features
attention_features_shape = 64   # dimensions of each feature

# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap

# Slice the images and captions into a new Dataset tensor
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

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
#for elem in dataset:
#    print(elem)
#    print("\n")

# Create the CNN_Encoderddaaa
encoder = CNN_Encoder(embedding)
decoder = RNN_Decoder(embedding, units, vocab_size, BATCH_SIZE)

# Create the loss functions for cross entropy
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

# Checkpoint for saving weights to the directory
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

# Training
# Steps:
#   1. Extract features stored in respective .npy files (pass features
#       through the encoder)
#   2. Encoder output, hidden state (init to 0) and decoder input (start token)
#       is passed to the docder
#   3. Decoder returns predictions and decoder hidden state
#   4. Docoder hidden state is then passed back into the model and the
#       predictions are used to calculate the loss.
#   5. Use teacher forcing to decide the next input to the decoder (GT)
#   6. Teacher forcing - technique where the target word is passed as the next
#       input to the decoder.
#   7. Final step is to calculate gradients and pply optimizer and backprop
    
# training cell will be reset if many runs are done (create new list)
loss_plot = []

#@tf.function
def train_step(img_tensor, target):
    loss = 0
    
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * 
                               target.shape[0], 1)
    
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            
            loss += loss_function(target[:, i], predictions)
            
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    
    total_loss = (loss / int(target.shape[1]))
    
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    
    gradients = tape.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return loss, total_loss

EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    
    for(batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        
        # show the batch loss for every 100 batches
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
            
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    # saving (checkpoint) the model every 5 epochs
    if epoch % 5 == 0:
        ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
# Plot the loss function
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
    
# Captioning
# Steps:
#   1. Evaluate function is similar to training loop, except no teacher
#       forcing is used. The input to the decoder at each time step is its
#       previous predictions along with hidden state and encoder output
#   2. Stop predicting when the model predicts end token
#   3. Store attention weights for every time step
def evaluate(image):
    attention_plot = np.zeros((max_len, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_len):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    len_result = len(result)
    print("Plot Attention:")
    print("len result = ", len_result)
    print(result)
    print("\n")
    if (len_result > 1):
        fig = plt.figure(figsize=(10, 10))
        total_dim = len_result//2 * len_result//2
        print("Total dim = ", total_dim)

        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            print("l + 1 = ", (l+1))
            if ((l+1) > total_dim):
                print("THIS WILL PASS AN ERROR!")
            else:
                print("THIS WILL NOT PASS ERROR:")
                ax = fig.add_subplot(len_result//2, len_result//2, l+1)
                ax.set_title(result[l])
                img = ax.imshow(temp_image)
                ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    else:
        temp_att = np.resize(attention_plot[0], (8, 8))
        fig, ax = plt.subplots()
        ax.set_title(result[0])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())


    plt.tight_layout()
    plt.show()

# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print("-------------------------------------------------------")
print("Final result:")
print("len result = ", len(result))
print(result)
print("\n")

print ('Real Caption:       ', real_caption)
print ('Prediction Caption: ', ' '.join(result))
print ("image = ", image)
print("\n")
plot_attention(image, result, attention_plot)
print("-------------------------------------------------------")
print("\n")
print("\n")


# Try your own images (Surf and turf)
image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension,
                                     origin=image_url)

print("Test Image:")
result, attention_plot = evaluate(image_path)
print ('Test Prediction Caption:', ' '.join(result))
plot_attention(image_path, result, attention_plot)
print("-------------------------------------------------------")

print("\n")
print("\n")

# opening the image
#Image.open(image_path)
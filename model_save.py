#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:32:49 2019

@author: alejandrogonzales
"""
# Saving model weights and architecture (exmaple file)

# Load weights from external .h5 model using TF
# Procedure:
#   1. First build the model with the same architecture as the original (see checkpoint)
#   2. Use the Model load weights from the .h5 model file
'''
new_model = build_model(
        data_size = data_size,
        embedding_dim = embedding_dim,
        rnn_units = rnn_units,
        batch_size = BATCH_SIZE)
# test the untrained model (for comparison with weights model)
new_model.evaluate(x_test, y_test)
new_model.load_weights('~/.keras/model/*.h5')   # load the weights only
load_model('~/.keras/model/.h5')    # load mode from h5 file
new_model.evaluate(x_test, y_test)  # cross-ref w untrained
'''

# Model fit will save the weights and params from the training (however the fit is done)
# names of checkpoint files (saves the model and params as the model is trained)
'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
'''
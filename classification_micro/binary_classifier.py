#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:45:10 2019
@author: tati
"""
# uncomment if checkpoints are needed
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import binary_utils as ub

#%% input generators & model skeleton
train, test = ub.create_generators(IMAGE_WIDTH = 120,IMAGE_HEIGHT = 120)
model = ub.model_binary()

# optional, can be removed. See keras documentation
#earlystop = EarlyStopping(patience=10)
#learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
#                                            patience=2, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0.00001)
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
#callbacks = [learning_rate_reduction, checkpoint]
#%%
# train the model
history = model.fit_generator(
    train,
    steps_per_epoch=300,
    epochs=50,
    shuffle=False)

#%% test it

# in case you have to start a new session, uncomment
#model = ub.model_binary()
#model.load_weights('weights_binary.hdf5')
num_of_test_files = 23
probabilities = model.predict_generator(test, num_of_test_files)
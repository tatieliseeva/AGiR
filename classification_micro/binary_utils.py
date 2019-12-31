#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:45:13 2019

@author: tati

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
from os import listdir, path, makedirs
import requests
from glob import glob
from os.path import join, exists  
import random
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

    
def plot_it_all(images_path, title = None):
    """plot all images in directory
    IN: picture name as str, title"""
        
    picfiles = glob(images_path)
    
    for pic_name in picfiles:
        img=mpimg.imread(pic_name)
        plt.figure()
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.imshow(img)       
    return

def download_it(list_of_links, folder_name = "name/"):   
    """download images from online sources
    IN: list of image links
    OUT: 
    """
      
    names = list_of_links[0]
    number = 0
    
    if not exists(folder_name):
        makedirs(folder_name)
        
    for banch_link in list_of_links[1:]:
        for link in banch_link:
            response = requests.get(link)
            if response.status_code == 200:
                number += 1
                file_name = join(folder_name + names +'_'+str(number)+'.jpg')
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                
    return

def visualize_random_images(folder_name = "dataset_barcoded", title = "barcoded",
                            number_of_random_images = 50):
    """ visualize random images from a folder
    IN: name of folder, title for images, number of random images
    OUT:
        """

    list_of_random_images = []

    [list_of_random_images.append(random.choice(listdir(folder_name))) for i in range(number_of_random_images)] 

    columns = 5
    rows = 10
    fig=plt.figure(figsize=(30, 30))
    i = 1
    for pic_name in list_of_random_images:
        image_to_visualize = path.join(folder_name, pic_name)
        img=mpimg.imread(image_to_visualize)
        fig.add_subplot(rows, columns, i)
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.imshow(img) 
        i = i +1  

    return  

#############################################################################
#                            net utils
#############################################################################
    
def model_binary():
    """ conv net for binary classification
    IN:
    OUT: compiled model
    """

    
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (120, 120, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    classifier.add(BatchNormalization(axis = -1))
    classifier.add(Dropout(0.2))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    classifier.add(BatchNormalization(axis = -1))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(activation = 'relu', units=512))
    classifier.add(BatchNormalization(axis = -1))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(activation = 'relu', units=256))
    classifier.add(BatchNormalization(axis = -1))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(activation = 'sigmoid', units=2))
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    classifier.summary()
       
    return classifier


def create_generators(IMAGE_WIDTH, IMAGE_HEIGHT, validation=True):
    """ creates train, val and test generators for model. Default image size 512x512
    IN: 
    OUT: train, val and test generators
    """
    
    #IMAGE_WIDTH = 512
    #IMAGE_HEIGHT = 512
    BATCH_SIZE = 32
    
    training_data_dir = "/home/tati/Desktop/cheek_cells_micronuclei_13_12/train"
    test_data_dir = "/home/tati/Desktop/cheek_cells_micronuclei_13_12/test"
    
    training_data_generator = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=15,
        vertical_flip=True,
        fill_mode='reflect',
        data_format='channels_last',
        brightness_range=[0.5, 1.5],
        featurewise_center=True,
        featurewise_std_normalization=True)

       
    test_data_generator = ImageDataGenerator(rescale=1./255)
    
    train_gen = training_data_generator.flow_from_directory(
        training_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary")
  
    
    test_gen = test_data_generator.flow_from_directory(
        test_data_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1,
        class_mode="binary", 
        shuffle=False)

    
    return train_gen, test_gen          
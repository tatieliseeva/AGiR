# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:40:41 2019

@author: tatiana

§ document.querySelectorAll("div.input").forEach(function(a){a.remove()}) ß remove code from generated html

labels1 = felzenszwalb(img, scale=100.0, sigma=0.98, min_size=200) also good for cometes

"""
import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from sklearn import cluster

from skimage import io, feature, filters, color
from skimage.morphology import square, dilation, watershed, remove_small_objects
from skimage.measure import label
from skimage.segmentation import  felzenszwalb


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """
    def _store(x):
        lst.append(np.copy(x))
    return _store

def edge_detection(picture, sigma=0.3, low_threshold=0.1, high_threshold=0.25):
    """ IN: picture path
        OUT: 2D array with detected edges
    """
    img = io.imread(picture, as_gray=True)
    img = feature.canny(img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    img = ndi.binary_fill_holes(img)  
    
    return img.astype(int)

# clusterisation is good for the comets
def clusterisation_kmeans (img, n_clusters=2, u_cometes=False):
    """ IN: picture path
        OUT: 3D array clustered
    """
    x, y, z = img.shape
    image_2d = img.reshape(x*y, z)
    image_2d.shape
    kmeans_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=6, 
                                    max_iter=300, tol=0.0001, algorithm='full')
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    clust = cluster_centers[cluster_labels].reshape(x, y, z)
    
    if u_cometes != False:
        unique, counts = np.unique(clust, return_counts=True)    
        # transfer to binary mask
        for x in np.nditer(clust, op_flags=['readwrite']):
            if x[...] != min (unique): 
                x[...] = 20
            else:
                x[...] = 180

    return clust

def edge_detection_custom(img):
    """ edge detection micro images
        IN: image as 3D array
        OUT: 2Dobject mask
    """
    edged = edge_detection(img)
    dilated = dilation(edged, square(10))
    dilated = ndi.binary_fill_holes(dilated).astype(int)
    dilated = dilation(dilated, square(3))
    dilated = ndi.binary_fill_holes(dilated).astype(int)
    dilated = label(dilated)
    dilated = remove_small_objects(dilated, min_size = 20000)
    
    return dilated

def sobel_custom(xx):
    """ uses sobel segmentation, setup for cell  detection
        IN: image as 3D array
        OUT: 2Dobject mask
    """
    
    img = mpimg.imread(xx) 
    img5 = filters.sobel(color.rgb2grey(img))
    markers = np.zeros_like(img5)
    markers[img5 < 0.060] = 0
    markers[img5 > 0.0150] = 1
        
    img6 = dilation(markers, square(2)).astype(int)
    img7 = ndi.binary_fill_holes(img6)
    img8 = label(img7)
    img9 = remove_small_objects(img8, min_size = 22000)
    
    blur = cv2.blur(img9,(5,5))
    
    return blur


def find_sells_watershed(mask):
    """ uses sobel segmentation, setup for cell  detection
        IN: 2Dobject mask
        OUT: 2Dobject mask
    """
    mask = remove_small_objects(mask, min_size = 10000)
    distance = ndi.distance_transform_edt(mask)
    local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((25, 25)),
                        labels=mask)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_objects(labels, min_size = 20000)     
    
    return labels

# find masks for cells in microscope
def find_masks_micro (files):
    i = 0
    for xx in files:
            
        img = mpimg.imread(xx)
              
        img2 = sobel_custom(xx) # more smooth than edge_detection    
        img32 = find_sells_watershed(img2)     
        labels1 = felzenszwalb(img, scale=100.0, sigma=0.98, min_size=11000)
        
        fig, axes = plt.subplots(1, 3, figsize=(19.20,10.80))
        ax = axes.flatten()
        
        ax[0].imshow(img)
        ax[0].set_axis_off()
        ax[0].set_title("Original", fontsize=12)
    
        ax[1].imshow(img)
        ax[1].contour(img32, colors='r')
        ax[1].set_axis_off()
        ax[1].set_title("Sobel", fontsize=12)
        
        ax[2].imshow(img)
        ax[2].contour(labels1, colors='r')
        ax[2].set_axis_off()
        ax[2].set_title("Felzenswalb", fontsize=12)
        
       
        plt.savefig(str(i)+'-Sobel-Felzenszwalb.png')
        plt.show()
        i = i + 1
    
    return ()   


if __name__ == "__main__":
    cwd = os.getcwd()
    pictures = []    
    [pictures.append(filename) for filename in Path(cwd).rglob('*.png') and  Path(cwd).rglob('*.jpg')]
    find_masks_micro(pictures[4:21])
   
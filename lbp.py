# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:29:10 2021

@author: Mahsa
"""

import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
# from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from skimage import io, exposure
import os
from PIL import Image
from scipy.spatial.distance import euclidean

def lbp_histogram(color_image):
    img = color.rgb2gray(color_image)
    patterns = local_binary_pattern(img, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return patterns


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    # z_norm = norm(diff.ravel(), 0)  # Zero norm
    return m_norm

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
    
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

import csv
img_path = r"C:\Users\Mahsa\Desktop\Tasks\7-#P3105-CV-Matlab\dataset\train\1" # Enter Directory of all images  


folder = img_path
images = [os.path.join(root, filename)
          for root, dirs, files in os.walk(folder)
          for filename in files
          if filename.lower().endswith('.jpg')]

img_path_test = r"C:\Users\Mahsa\Desktop\Tasks\7-#P3105-CV-Matlab\dataset\test\1" # Enter Directory of all images  

folder_test = img_path_test
images_test = [os.path.join(root, filename)
          for root, dirs, files in os.walk(folder_test)
          for filename in files
          if filename.lower().endswith('.jpg')]

import itertools


dist_accuracy = []
count=0
# for image in images:
for image in itertools.islice(images , 0,40):
    src = io.imread(image)
    prediction = lbp_histogram(src)
    count +=1
    
    # img_float32_2 = np.float32(prediction)
    # destination_gray = cv2.cvtColor(img_float32_2, cv2.COLOR_RGB2HSV)
    # img1 = original_gray
    img2 =  np.expand_dims(prediction, axis=2)
    # Accuracy Compare prediction from patterns to test data
    for image_test in itertools.islice(images_test , 0, 40):
        src_test = cv2.imread(image_test)
        img_float32_1 = np.float32(src_test)
        original_gray = cv2.cvtColor(img_float32_1, cv2.COLOR_RGB2HSV)

        img1 = original_gray
        img2 = img2
        
        n_m = compare_images(img1, img2)
        dist_accuracy.append(n_m/img1.size)
    
        print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)




with open('lbp_sample_40.csv', 'w', encoding='UTF8') as f1:
    csvcreator_x = csv.writer(f1)
       
    csvcreator_x.writerow(dist_accuracy)
    
import statistics

avg_acc = statistics.mean(dist_accuracy)  # Average distances of all images
print("Average distance %d" %(avg_acc*(1/count)))

float_list = [count,avg_acc]
with open('lbp_sample_x.csv', 'a', newline='', encoding='UTF8') as f_wrapup:
    csvcreator_x = csv.writer(f_wrapup)
       
    csvcreator_x.writerow(float_list)
import os
import cv2
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

IMAGES = "extracted_faces/"

def pre_words_extraction(img):
    img = local_binary_pattern(img, 8, 3)
    pre_words = []
    step = 4
    for i in range(0, img.shape[0], step):
        for j in range(0, img.shape[1], step):
            pre_words.append(img[i*step:(i+1)*step, j*step:(j+1)*step])
    pre_words = map(lambda M: M.reshape((1, M.size))[0], pre_words)
    return pre_words


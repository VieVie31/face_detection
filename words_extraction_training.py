import os
import cv2
import sys
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
    #extract pre words from a LBP filterd image
    img = local_binary_pattern(img, 8, 3)
    pre_words = []
    step = 4
    for i in range(0, img.shape[0], step):
        for j in range(0, img.shape[1], step):
            pre_words.append(img[i:i+step, j:j+step])
    pre_words = map(lambda M: M.reshape((1, M.size))[0], pre_words)
    return pre_words

def get_translator(pre_words, nb_words=200):
    k = KMeans(n_clusters=nb_words)
    k.fit(pre_words)
    return k

def get_words(img, translator):
    #converts all prewords to words in the image
    pre_words = pre_words_extraction(img)
    words = translator.predict(pre_words)
    return words

if __name__ == '__main__':
    #extract pre words
    pre_words_extracted = []
    for image_name in os.listdir(IMAGES):
        try:
            pre_words_extracted += pre_words_extraction(imread(IMAGES + image_name))
        except:
            pass
    #making a pre words to words translator
    k = get_translator(pre_words_extracted, 200)
    #save the translator
    f = open("translator.mdl", "wb")
    pickle.dump(k, f)
    f.close()

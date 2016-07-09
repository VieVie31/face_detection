import os
import re
import cv2
import sys
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from lbp_face_detection_model_training import *


def extract(filename, clf, keep=1):
    lst = []
    image = cv2.imread(filename, 0)
    for i, image in enumerate(pyramid(image, min_size=64, step=.75)):
        #cv2.imshow("image", image)
        for (x, y, img) in sliding_window(image, 8, (64, 64)):
            if img.shape != (64, 64):
                continue
            feat = pre_processing(img)
            if clf.predict([feat])[0] == 1:
                #lst.append(img)
                lst.append((clf.predict_proba([feat])[0,1], img))
    if len(lst) == 0:
        return []
    lst = sorted(lst)[::-1]
    lst = lst[:keep]
    return zip(*lst)[1]
    #return lst

if __name__ == '__main__':
    #load the model
    f = open("logistic_model.mdl", "rb")
    clf = pickle.load(f)
    f.close()

    #extract faces of the file given in param
    img_file = sys.argv[1]
    lst = extract(img_file, clf)

    #save extracted faces
    for i, v in enumerate(lst):
        cv2.imwrite(str(i) + "_" + str(time.time()) + ".jpg", v)



    

import os
import re
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

def imread(filename):
    if filename[:-4] == 'pgm':
        return read_pgm(filename)
    else:
        return cv2.imread(filename, 0)

def normalize(t):
    return (t - t.mean()) / t.std()

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, min_size=64, step=0.75):
    w, h = image.shape
    yield image
    while min(w, h) > min_size:
        w, h = image.shape
        image = cv2.resize(image, (int(h * step), int(w * step)))
        yield image

def distance(a, b):
    return sum((a - b)**2) ** .5


def random_split(dataset, training_proportion):
    random.shuffle(dataset)
    return (
        dataset[:int(training_proportion * len(dataset))],
        dataset[int(training_proportion * len(dataset)):])

def hist_256(t):
    hist = [0] * 256
    for v in t:
        hist[int(v)] += 1
    return hist

def shuffled(lst):
    random.shuffle(lst)
    return lst

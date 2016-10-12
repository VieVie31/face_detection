import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.filter import sobel
from sklearn.cluster import KMeans


def binarize_img(img):
    #binarize image with k means (k=2)
    k = KMeans(n_clusters=2)
    k.fit(img.reshape((64 * 64, 1)))

    binarized_img = k.predict(img.reshape((64 * 64, 1)))
    binarized_img = binarized_img.reshape((64, 64))

    v = binarized_img.sum(axis=1)
    h = binarized_img.sum(axis=0)

    n = np.array([h] * 64)

    for i in range(64):
        n[:,i] += v

    binarized_img -= 1
    binarized_img = abs(binarized_img)
    return binarized_img

def eyes_detection(binarized_img):
    #as the parts of hair in the image can move significantly the centroid
    #the use of sobel filter to get edges on binary images could mitigate this issue
    sob = sobel(binarized_img) > 0

    #split the image in 4 parts
    top_left  = sob[:32,:32]
    top_right = sob[:32,32:]
    bot_left  = sob[32:,:32]
    bot_right = sob[32:,32:]

    #get the top_left cluster center
    x, y = np.where(top_left == 1)
    x, y = x.tolist(), y.tolist()

    k = KMeans(n_clusters=1)
    k.fit(zip(x, y))
    top_left_center = k.cluster_centers_[0,:]

    #get the top_right cluster center
    x, y = np.where(top_right == 1)
    x, y = x.tolist(), y.tolist()

    k = KMeans(n_clusters=1)
    k.fit(zip(x, y))
    top_right_center = k.cluster_centers_[0,:]

    #convert the relatives cluster centers to absolute cluster center
    #the eyes should be here... or something close...
    top_left_center  = np.rint(top_left_center).tolist()
    top_right_center = np.rint(top_right_center).tolist()
    top_right_center[1] += 32

    top_left_center  = int(top_left_center[0]),  int(top_left_center[1])
    top_right_center = int(top_right_center[0]), int(top_right_center[1])

    return top_left_center, top_right_center


def angle(p1, p2):
    a = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    return np.rad2deg(a)

def get_face_rotation_angle(img):
    bi = binarize_img(img)
    top_left_center, top_right_center = eyes_detection(bi)
    theta = angle(top_left_center, top_right_center)
    return theta

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)

    #binarization of the image & eyes detection
    print "binarization of the image..."
    bi = binarize_img(img)
    
    print "eyes detection..."
    top_left_center, top_right_center = eyes_detection(bi)

    bi[top_left_center[0],  top_left_center[1]]  = 5
    bi[top_right_center[0], top_right_center[1]] = 5
    plt.imshow(bi)
    plt.show()


    #get face rotation angle
    print "getting the face rotation angle..."
    theta = angle(top_left_center, top_right_center)
    print top_left_center
    print top_right_center
    print "rotation angle : {}".format(theta)

    #rotate the image to align face... seems to be not really performant... :'(
    M = cv2.getRotationMatrix2D((32, 32), theta, 1)
    dst = cv2.warpAffine(img, M, (64, 64), borderValue=img.mean())

    #show the rotated image
    plt.imshow(dst)
    plt.show()

    #save the rotated image
    cv2.imwrite(sys.argv[1] + "_rotated.jpg", dst)



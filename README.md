# Face Detection

This repository deals with some stuffs about face detection...

## Requirements

It requires python 2.7 with the following modules :

```
cv2        --> opencv
numpy      --> numerical computation
skimage    --> for computing lbp efficiently
sklearn    --> for machine learning
matplotlib --> for plotting and visualizing
```

## Files

### tools.py

Some usefull functions

### lbp_face_detection_model_training.py

Train a Logistic Regression model to learn to detect faces using the lbp features...
At the end of the training the model is saved as logistic_model.mdl with help th the pickle module...
To train the model :
```bash
python lbp_face_detection_model_training.py
```

### extract.py

Extract faces from the image... this can take a while because no heuristic was made actually and does a lot of matrix multiplcations...
To run it type (once the lbp_model trained) : 
```bash
python extract.py my_image_name.jpg
```

### eyes_detection.py

Try to find where are eyes in the detected face...
The image is first binarized using the k-means algorithm (k=2) to reveal the darkest parts of the images (eyes, mouth, hairs...)

![binarized image using kmean](img/k_mean_2k.png "binarized image using k-means")

A sobel operator is applyed on the binaried image to improve the accuracy of detecting eyes... In did, without this trick, the center of a the cluster was biaised by the hairs if there was any on the picture, or other shadow, but now this issue seems to be solved by applying this operator and using only the edges found as points...

Look at the difference :

Without the sobel trick :
![without sobel operator](img/eyes_center_with_binarized_image.png "without sobel operator")
![without sobel operator 2](img/without_sobel.png "without sobel")

And now with the sobel trick :
![with sobel operator trick](img/eyes_center_with_sobel_on_binarized_image.png "with the sobel operato trick")
![with sobel operator trick 2](img/with_sobel.png "with the sobel trick")

The sobel edged binarized image is splitted in 4 parts, 2 tops and 2 bottoms, and with 1 cluster k-means on each of the top image, the eyes should be near to the center cluster...

### face aligment

After eyes localisaiton, we can perform aligment of the face using the angle of the vector formed by the 2 points : the eyes...

croped face from an image :
![face detected not aligned](img/not_aligned_face.png "face detected not aligned")
detection of the eyes :
![eyes found](img/eyes_detected.png "eyes found")
same face after eyes aligment :
![aligned](img/eyes_aligned.png "eyes aligment")


```bash
python eyes_detection.py my_extracted_face.jpg
```

# Video Demo
On youtube :

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-6W8UtYoLOE/0.jpg)]
(https://www.youtube.com/watch?v=-6W8UtYoLOE)



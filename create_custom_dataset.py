import numpy as np
from sklearn.datasets import fetch_openml
import cv2 as cv
import matplotlib.pyplot as plt
import os

images = []
labels = []
ad = r"G:\jadi\DATA\mnist\digits"

for file in os.listdir(ad):
    img = cv.imread(os.path.join(ad, file), cv.IMREAD_GRAYSCALE)

    images.append(img.ravel())
    labels.append(int(file[0]))

images = np.array(images)
labels = np.array(labels)
np.savez_compressed("digits.npz", images=images, targets=labels)


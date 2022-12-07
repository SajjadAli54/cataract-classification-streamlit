import numpy as np
import pandas as pd
import cv2 as cv
import streamlit as st
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import joblib

indextable = ['dissimilarity', 'contrast',
              'homogeneity', 'energy', 'correlation', 'Label']
obj = {
    0.0: "Normal",
    1.0: "Cataract",
    2.0: "Glaucoma",
    3.0: 'Retina Disease'
}
width, height = 400, 400
distance = 10
teta = 90

# Code to extract features from Image using Gray Level Co occurrence Image


def get_feature(matrix, name):
    feature = graycoprops(matrix, name)
    result = np.average(feature)
    return result


def preprocessingImage(image):
    test_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img_thresh = cv.adaptiveThreshold(
        test_img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3)

    cnts = cv.findContours(
        test_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        test_img_ROI = test_img[y:y+h, x:x+w]
        break

    test_img_ROI_resize = cv.resize(test_img_ROI, (width, height))
    test_img_ROI_resize_gray = cv.cvtColor(
        test_img_ROI_resize, cv.COLOR_RGB2GRAY)

    return test_img_ROI_resize_gray


def extract(path):
    data_eye = np.zeros((5, 1))

    # path = cv.imread(path)
    img = preprocessingImage(path)

    glcm = graycomatrix(img, [distance], [teta],
                        levels=256, symmetric=True, normed=True)

    for i in range(len(indextable[:-1])):
        features = []
        feature = get_feature(glcm, indextable[i])
        features.append(feature)
        data_eye[i, 0] = features[0]
    return pd.DataFrame(np.transpose(data_eye), columns=indextable[:-1])


"""
Return predicted class with its probability
"""

model = joblib.load("model.pkl")


@st.cache
def predict(path):
    X = extract(path)
    y = model.predict(X)[0]
    prob = model.predict_proba(X)[0, int(y)]
    return (obj[y], prob)

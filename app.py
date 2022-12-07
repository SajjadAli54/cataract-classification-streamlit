import streamlit as st
from PIL import Image
from models import predict
import numpy


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.title("Cataract Image Classification")

st.header('Enter the fundus image')
st.subheader("KNN Model")

image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:
    img = load_image(image_file)
    st.image(img, width=250)
    open_cv_image = numpy.array(img)
    label, prob = predict(open_cv_image)
    st.write(f"Label : {label}")
    st.write(f"Probability: {prob}")

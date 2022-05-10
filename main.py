import streamlit as st
from streamlit_option_menu import option_menu

import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import load_model
@st.cache(allow_output_mutation=True)


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #prediction_percentage = model.predict(data)
    #prediction = prediction_percentage.round()
    return prediction

    #return prediction, prediction_percentage

def demo():
    st.markdown("Try")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded file', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        class_names = ['Giveway', 'NoEntry', 'NoHorn', 'Roundabout', 'Stop']
        Ans = teachable_machine_classification(image, 'Inception_DA.h5')
        string=class_names[np.argmax(Ans)]
        st.header(string)
        #df = pd.DataFrame(label, columns=['Giveway', 'NoEntry', 'NoHorn', 'Roundabout', 'Stop'])
        #st.write(df)


def mm():
    st.title("ကျေးဇူးပြု၍ မြန်မာဘာသာစကားအတွက် ဤနေရာကိုနိုပ်ပါ။")
    st.subheader("https://share.streamlit.io/aungmoekyaw135/traffic-sign-detection/main/burmese.py")


def main():
    st.title("Traffic sign detection")
    st.subheader("from the mind of Team ZAC.")

    with st.sidebar:
        selected = option_menu("Main Menu", ["Demo", 'Info'],
                               icons=['house', 'info-circle'], menu_icon="cast", default_index=1)

        st.subheader("💬Language")
        if selected == "Demo" or "Info":
            selected0 = option_menu(None, ["English", "မြန်မာ"],
                                    icons=['fgngf', "translate"], orientation="horizontal")
            st.text("©2022_Team_ZAC")
    if selected0 == "မြန်မာ":
        mm()


    if selected == "Info":
        selected1 = option_menu(None, ["Term and con", "Developer contact" ],
                            icons=['clipboard', 'chat-right-dots'],
                            menu_icon="cast", default_index=0, orientation="horizontal")

        if selected1 == "Term and con":
            st.header("Term and Condition")
            st.write("Please read the terms and cons before using our model")
            st.write("")
            st.write("Although we consider unexpected circumstances, there are some weaknesses that we can't control such as the camera error, and camera view-blocking by trash on the road. We do committed to evolving development.")
            st.write("")
            st.write("By using our model, we assume you read all the terms and conditions and are aware of the weakness, such as having a large rain camera isn't clear to detect. It's intuitively clear to acknowledge that our models are unable to let it alone without any humanoid supervisor.")
            st.write("...More....")
        
        if selected1 == "Developer contact":
            st.write("noname@gmail.com")
            st.write(".....More.....")
        
    if selected == "Demo":
        selected0 = option_menu(None, ["Demonstration", "Evaluation Metric", ],
                            icons=['activity', "check2-circle"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
        
    if selected0 == "Demonstration":
        
        demo()
    if selected0 == "Evaluation Metric":
        st.image("img.jpg")


main()

import streamlit as st
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


@st.cache(allow_output_mutation=True)

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    
    return  prediction,prediction_percentage


    
def mm():
    st.title("ကျေးဇူးပြု၍ မြန်မာဘာသာစကားအတွက် ဤနေရာကိုနိုပ်ပါ။")
    st.subheader("https://share.streamlit.io/zinwaiyan274/zac/main/burmese.py")


st.title("Traffic sign detection")
st.subheader("from the mind of Team ZAC.")
    
with st.sidebar:
    selected = option_menu("Main Menu", ["Demo", 'Info'],
                               icons=['house', 'info-circle'], menu_icon="cast", default_index=1)

    st.subheader("Language")
    # selected0 = option_menu(None, ["Language"],
    #                         icons=["translate"], orientation="horizontal")

    icons=["translate"]
    if selected == "Demo" or "Info":
        selected0 = option_menu(None, ["English", "မြန်မာ"],
                                    icons=['spellcheck', "translate"], orientation="horizontal")

        st.text("©2022_Team_ZAC")
if selected0 == "မြန်မာ":
    mm()

if selected == "Info":
    selected1 = option_menu(None, ["Term and con", "Developer contact",  ],
                            icons=['clipboard', 'chat-right-dots', "flag fill"],
                            menu_icon="cast", default_index=0, orientation="horizontal")

if selected == "Demo":
    selected2 = option_menu(None, ["Demonstration", "Computer Vision", " Evaluation Metric", ],
                            icons=['activity', 'eye-fill', "check2-circle"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
if selected2 == "Demonstration":
    uploaded_file = st.file_uploader("Choose a Image...")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded file', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label, perc = teachable_machine_classification(image, 'Final.h5')
        # st.write(label)
        df = pd.DataFrame(label, columns=['Giveway', 'NoEntry', 'NoHorn', 'Roundabout', 'Stop'])
        st.subheader(df)



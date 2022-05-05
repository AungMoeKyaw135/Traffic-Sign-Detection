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
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction = prediction_percentage.round()

    return prediction, prediction_percentage


def demo():
    st.markdown("ကြိုဆိုပါတယ်")
    uploaded_file = st.file_uploader("ပုံရွေးရန်")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded file', use_column_width=True)
        st.write("ခန့်မှန်းချက်"/n/n/n)
#         st.sub("")
        class_names = ['ဦးစာပေးစေ၍ ဖြည်းဖြည်းမောင်းပါ', 'မဖြတ်သန်းရ', 'ဟွန်းမတီးရ', 'လှည့်သွားပါ', 'ရပ်']
        Ans = teachable_machine_classification(image, 'Final.h5')
        string = class_names[np.argmax(Ans)]
        string = st.header(string)
        st.write('ပုံ၏ အဓိပ္ပာယ်မှာ - ', string)
        # df = pd.DataFrame(label, columns=['Giveway', 'NoEntry', 'NoHorn', 'Roundabout', 'Stop'])
        # st.write(df)



def eng():
    st.title("ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာစကားအတွက် ဤနေရာကိုနိုပ်ပါ")
    st.subheader("https://share.streamlit.io/zinwaiyan274/zac/main/English.py")



def main():
    st.title("Traffic sign detection")
    st.subheader("from the mind of Team ZAC.")

    with st.sidebar:

        selected = option_menu("ပင်မစာမျက်နှာ", ["လုပ်ဆောင်ချက်", "အကြောင်းအရာ"], icons=['house', 'info-circle'], menu_icon="cast", default_index=1)


        st.subheader("ဘာသာစကား")
        if selected == "လုပ်ဆောင်ချက်" or "အကြောင်းအရာ":
            selected0 = option_menu(None, ["မြန်မာ", "အင်္ဂလိပ်"], orientation="horizontal")

            st.text("©2022_Team_ZAC")
    if selected0 == "အင်္ဂလိပ်":
        eng()
    if selected == "အကြောင်းအရာ":
        selected1 = option_menu(None, ["စာချုပ်", "တီထွင်သူများသို့ ဆက်သွယ်ခြင်း"], icons=['clipboard', 'chat-right-dots'],
                            menu_icon="cast", default_index=0,orientation="horizontal")

    if selected == "လုပ်ဆောင်ချက်":
        selected0 = option_menu(None, ["လုပ်ဆောင်ချက်", "ကင်မရာ", "ရလဒ်"], orientation="horizontal")
        if selected0 == "လုပ်ဆောင်ချက်":
            demo()



st.text("©2022_Team_ZAC")

main()

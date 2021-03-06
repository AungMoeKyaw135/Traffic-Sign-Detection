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
    prediction_percentage = model.predict(data)
    prediction = prediction_percentage.round()

    return prediction, prediction_percentage


def demo():
    st.markdown("ကြိုဆိုပါတယ်")
    uploaded_file = st.file_uploader("ပုံရွေးရန်")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded file', use_column_width=True)
        st.write("ခန့်မှန်းလိုက်သောပုံ၏ အဓိပ္ပာယ်မှာ - ")
#         st.sub("")
        class_names = ['ဦးစာပေးစေ၍ ဖြည်းဖြည်းမောင်းပါ', "မဖြတ်သန်းရ", 'ဟွန်းမတီးရ', 'လှည့်သွားပါ', 'ရပ်']
        Ans = teachable_machine_classification(image, 'Inception_DA.h5')
        string = class_names[np.argmax(Ans)]
        st.header(string)
#         st.write('ပုံ၏ အဓိပ္ပာယ်မှာ - ', string)
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
        if selected1 == "စာချုပ်":
            st.header("Term and Condition")
            st.write("ကျွန်ုပ်တို့၏ မော်ဒယ်ကို အသုံးမပြုမီ စည်းကမ်းသတ်မှတ်ချက်များကို ဖတ်ပါ။")
            st.write("")
            st.write("မမျှော်လင့်ထားသော အခြေအနေများကို ကျွန်ုပ်တို့စဉ်းစားသော်လည်း၊ ကင်မရာအမှားအယွင်းနှင့် လမ်းပေါ်ရှိအမှိုက်များဖြင့် ကင်မရာမြင်ကွင်းပိတ်ဆို့ခြင်းကဲ့သို့သော အားနည်းချက်အချို့ရှိပါသည်။ ကျွန်ုပ်တို့သည် ဖွံ့ ဖြိုးတိုးတက်မှုအတွက် ကတိပြုပါသည်။")
            st.write("")
            #st.write(" ကျွန်ုပ်တို့၏ မော်ဒယ်ကို အသုံးပြုခြင်းဖြင့် သင်သည် စည်းကမ်းသတ်မှတ်ချက်များအားလုံးကို ဖတ်ရှုပြီး ကင်မရာတွင် မိုးကြီးနေပါက ထောက်လှမ်းရန် မရှင်းလင်းခြင်းကဲ့သို့သော အားနည်းချက်များကို သိရှိသည်ဟု ကျွန်ုပ်တို့ ယူဆပါသည်။ ကျွန်ုပ်တို့၏ "Ai မော်ဒယ်" သည် လူသားကြီးကြပ်ရေးမှူးမပါဘဲ ၎င်းကို တစ်ယောက်တည်းထား၍မရကြောင်း ရှင်းရှင်းလင်းလင်း ")
            st.write("...More....")
        
        if selected1 == "တီထွင်သူများသို့ ဆက်သွယ်ခြင်း":
            st.write("Machine Learning / AI engineers : Zin Wai Yan, Chan Myae Oo, Aung Moe Kyaw")
            st.write("zinwaiyan274@gmail.com")
            st.write(".....More.....")

    if selected == "လုပ်ဆောင်ချက်":
    
        selected0 = option_menu(None, ["လုပ်ဆောင်ချက်", "ရလဒ်" ],
                            icons=['activity', "check2-circle"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    
        if selected0 == "လုပ်ဆောင်ချက်":
            demo()
        
        if selected0 == "ရလဒ်":
            st.image("img.png")



st.text("©2022_Team_ZAC")

main()

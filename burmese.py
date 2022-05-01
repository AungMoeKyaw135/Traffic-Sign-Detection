import streamlit as st
from streamlit_option_menu import option_menu

def demo():
    st.markdown("ကြိုဆိုပါတယ်")
    uploaded_file = st.file_uploader("ပုံရွေးရန်")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        st.write('ခန့်မှန်းမည့်ပုံမှာ အောက်ပါပုံဖြစ်သည်')
        st.image(uploaded_file)
        if st.button('ခန့်မှန်းမယ်'):
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)


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

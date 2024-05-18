import streamlit as st
from streamlit_option_menu import option_menu
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from skimage.io import imread


model = tf.keras.models.load_model('sbcnn.h5')('sbcnn.h5')
categories=['Volleyball', 'Tennis ball', 'Football', 'Cricket ball', 'Basketball']
st.set_page_config(layout="wide")
st.title(":red[SPORTS BALL TYPE PREDICTION ]")




st.write("  ")
st.write("  ")
selact=option_menu(menu_title=None,
                            options=["Home","Prediction"],
                            icons=["house","person-workspace"],
                            orientation="horizontal",
                        )

if selact == "Home":
    st.write("  ")
    st.write("  ")
    st.write("  ")
    st.image('1-15050Q93A7.jpg',width=400)


    project_description = """
            <div style="font-family: 'Arial Black', sans-serif;">
            <p style="font-size: 20px;"><b>Project Description</b></p>

            <p> Our project focuses on the development of a Convolutional Neural Network (CNN) model tailored specifically for the recognition of various sports balls.
             The dataset utilized for training and evaluation encompasses images of volleyball, tennis ball, football, cricket ball, and basketball.</p>
            </div>
            """
    st.markdown(project_description, unsafe_allow_html=True)
else:
    st.warning(
        "This prediction functionality only works for the following sports balls: Volleyball, Tennis Ball, Football, Cricket Ball, Basketball.")
    text_colum,image_colum=st.columns((1,1))

    with text_colum:

        uploaded_file = st.file_uploader("Choose a file",type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.write("File uploaded successfully!")

            with image_colum:
                st.image(uploaded_file, caption='Uploaded Image', width=300)

        pred=st.button("Identify Ball")
        if pred:

            def predict(img):
                image = imread(img)
                rimg = resize(image, ( 123, 123, 3))
                img=rimg.reshape(1,123,123,3)
                y_new = model.predict(img)
                i = y_new.argmax()

                return categories[i]


            prediction= predict(uploaded_file)
            st.write("### Ball Type :",prediction)



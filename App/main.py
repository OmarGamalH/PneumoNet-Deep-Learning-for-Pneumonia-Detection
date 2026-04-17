import streamlit as st
from io import BytesIO
import torch 
import numpy as np
import os 
from Utilities.Utilities import *
from PIL import Image

# Settings and model upload
st.set_page_config("Home")
current_dir = os.path.dirname(__file__)
assets_dir = os.path.join(current_dir , 'assets')
images_dir = os.path.join(current_dir , "Images")
upload_css_file(assets_dir=assets_dir , file_name = "styles.css")
model = load_model()
model = Model(model)
container_index = 0


# title and description
st.title("Pneumonia Classification")

# Slider
with st.sidebar:
    st.header("Settings")
    show_model_details = st.checkbox("Show Model Details" , value = False)
    run_model = st.checkbox("Run Model" , value = True)
    show_info = st.checkbox("Show Info" , value = False)

# Section : 1
if show_model_details:
    st.write("deep learning-based medical image classification system that detects pneumonia from chest X-ray images.")
    st.divider()


    image_1 = os.path.join(images_dir , "accuracy_loss.png")
    image_2 = os.path.join(images_dir , "predictions.png")
    
    st.image(image_1)
    with st.container(key = "description_0"):
        st.write("Shows how error decreases over epochs" )
    st.image(image_2)
    with st.container(key = "description_1"):
        st.write("Shows accuracy improvement over epochs")




st.divider()


# Section : 2
if run_model:
    images_files = st.file_uploader("Image Upload" , type = ['jpg' , 'jpeg' , 'png'] , accept_multiple_files=True , key = "upload_box" )
    st.divider()

    for file in images_files:
        file_bytes = file.getvalue()

        file_bytes_io = BytesIO(file_bytes)

        PIL_image = Image.open(file_bytes_io)

        transformed_image_tensor = model.transform(PIL_image)
        transformed_image_numpy = reorder_channels(np.array(transformed_image_tensor))

        
        col1 , col2 = st.columns(2)

        with col1:
            st.write("Original Image")
            st.image(file)
        
        with col2:
            st.write('Transformed Image')
            st.image(transformed_image_numpy)
        
        if run_model:
            prediction = model.predict(transformed_image_tensor)
            with st.container(key = f"prediction_container_{container_index}"):
                st.write(f"{prediction}")
                container_index += 1

        st.divider()


if show_info:
    st.header("Info")
    st.write("""
            Author : Omar Gamal Hamed
            """)
    st.link_button('Github Repo' , "https://github.com/OmarGamalH/PneumoNet-Deep-Learning-for-Pneumonia-Detection")
    st.link_button('linkedin' , 'https://www.linkedin.com/in/omar-gamal-hamed/')

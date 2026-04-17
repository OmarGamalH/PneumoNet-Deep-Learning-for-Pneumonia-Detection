import streamlit as st
from io import BytesIO
import torch 
import numpy as np
import os 
from Utilities.Utilities import *
from PIL import Image

st.header("Pneumonia Classifier")

current_dir = os.path.dirname(__file__)
assets_dir = os.path.join(current_dir , 'assets')




images_files = st.file_uploader("Image Upload" , type = ['jpg' , 'jpeg' , 'png'] , accept_multiple_files=True , key = "upload_box" )

model = load_model()
model = Model(model)

i = 0
with st.sidebar:
    st.header("Model Settings")
    run_model = st.checkbox("Run Model" , value = True)

upload_css_file(assets_dir=assets_dir , file_name = "styles.css")

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
        with st.container(key = f"prediction_container_{i}"):
            st.write(f"{prediction}")
            i += 1

    st.divider()

    


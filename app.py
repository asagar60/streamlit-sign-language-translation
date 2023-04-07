import cv2
import datetime, time
import os, sys
import numpy as np
from yolov5_utils import give_yolo_result, load_model

import glob
import streamlit as st
import wget
from PIL import Image
import torch
import random

st.set_page_config(layout="wide")

model = None
confidence = .25
device = 'cpu'

# ref : https://discuss.streamlit.io/t/deploy-yolov5-object-detection-on-streamlit/27675
# ref : https://github.com/thepbordin/YOLOv5-Streamlit-Deployment

def image_input(data_src):
    global confidence, model, device
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            # img = infer_image(img_file)
            # print(img_file)
            frame, txt = give_yolo_result(cv2.imread(img_file), model, size=640, confidence = confidence, device = device)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Model prediction")

def video_input(data_src):
    global confidence, model, device
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        output_text = ''
        op_textbox = st.image([])
        key = random.random()
        op_textbox.text_area("Output: ", "", height=100, key = key)
        while True:
            #time.sleep(0.3)
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame, txt = give_yolo_result(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), model, size=640, confidence = confidence, device = device)
            output.image(frame, channels = 'RGB', use_column_width = True)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            if txt != '':
                output_text += txt
                output_text += ' '
            del st.session_state[key]
            key = random.random()
            op_textbox.text_area("Output: ", output_text, height=100, key = key)
            

        cap.release()

def main():
    # global variables
    global confidence, model, device

    st.title("Live Sign Language Translation")

    st.sidebar.title("Settings")

    # try:
    # device options
    if torch.cuda.is_available():
        device = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    else:
        device = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # load model
    model = load_model(device)
    
    # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)
    
    st.sidebar.markdown("---")

    # input options
    input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

    # input src option
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

    if input_option == 'image':
        image_input(data_src)
    else:
        video_input(data_src)

    # except:
    #     st.warning("Model loading failed..! please added to the model folder.", icon="⚠️")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

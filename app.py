import streamlit as st
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import base64
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import io
import pims
import trackpy as tp
import streamlit_antd_components as sac

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Path to the saved model
model_path = "cell-segmentation/U-net-model.h5"

def resize_image(image, target_size=(128, 128)):
    # Resize the image to the target size and convert it to RGB mode
    return image.resize(target_size).convert("RGB")

# Function to preprocess test images
def preprocess_test_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = np.array(image)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    return img

def images():
    # Load the saved model
    model = load_model(model_path)

    # Display file uploader for test images
    st.title('Upload Test Images For Single Image Segmentation')
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        pred_masks = []
        uploaded_images = []
        
        # Create two columns for displaying uploaded images and predicted masks
        col3, col4 = st.columns(2)
        
        for uploaded_file in uploaded_files:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            uploaded_images.append(image)
            # Preprocess the image
            test_image = preprocess_test_image(image)
            
            # Expand the dimensions of the test image to create a batch of size 1
            test_image_batch = np.expand_dims(test_image, axis=0)

            # Predict segmentation mask
            pred_mask = model.predict(test_image_batch)

            # Threshold the predicted mask
            pred_mask_thresholded = (pred_mask > 0.5).astype(np.uint8)
            
            # Resize the predicted mask to match the height of the uploaded image
            pred_mask_resized = np.array(Image.fromarray(pred_mask_thresholded.squeeze()).resize(image.size))
            
            # Append the predicted mask to the list
            pred_masks.append(pred_mask_resized)
            
            # Display the uploaded image and predicted mask in the respective columns
            col3.image(image, caption='Uploaded Image', use_column_width=True)
            col4.image((pred_mask_resized * 255).astype(np.uint8), caption='Predicted Mask', use_column_width=True)




            

def video():
    # Load the saved model
    model = load_model(model_path)

    # Display file uploader for test images
    st.title('Upload Test Images For Multiple Image Segmentation')
    uploaded_files_v = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files_v:
        pred_masks = []
        uploaded_images = []
        
        for uploaded_file in uploaded_files_v:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            uploaded_images.append(image)
            # Preprocess the image
            test_image = preprocess_test_image(image)
            
            # Expand the dimensions of the test image to create a batch of size 1
            test_image_batch = np.expand_dims(test_image, axis=0)

            # Predict segmentation mask
            pred_mask = model.predict(test_image_batch)

            # Threshold the predicted mask
            pred_mask_thresholded = (pred_mask > 0.5).astype(np.uint8)
            
            # Append the predicted mask to the list
            pred_masks.append(pred_mask_thresholded.squeeze())
            
            # Append the uploaded image to the list
            

            # Plot the test image and predicted mask
            # st.image(image, caption='Uploaded Image', use_column_width=True)
            # st.image(pred_mask_thresholded.squeeze(), caption='Predicted Mask', use_column_width=True)
        
        def resize_image(image, target_size):
            if image is None:
                return None
            # Create a copy of the image array to ensure it owns its data
            image_copy = np.copy(image)
            # Convert the copy to PIL Image
            pil_image = Image.fromarray(image_copy)
            # Resize the image
            resized_image = pil_image.resize(target_size).convert("RGB")
            return resized_image

        # Define width and height for resizing
        width = 640  # Example width
        height = 480  # Example height
        target_size = (width, height)

        # Create an MP4 video combining uploaded images and predicted masks
        combined_video_path = 'combined_video.mp4'
        with imageio.get_writer(combined_video_path, fps=1) as writer:
            for img, pred_mask in zip(uploaded_images, pred_masks):
                # Resize the uploaded image and predicted mask
                resized_img = resize_image(img, target_size)
                resized_pred_mask = resize_image(pred_mask, target_size)

                # Check if either of the images is None
                if resized_img is None or resized_pred_mask is None:
                    continue

                # Convert images to numpy arrays
                img_array = np.array(resized_img)
                pred_mask_array = np.array(resized_pred_mask)

                # Scale image arrays to [0, 255] range
                img_scaled = (img_array * 255).astype(np.uint8)
                pred_mask_scaled = (pred_mask_array * 255).astype(np.uint8)

                # Combine images side by side
                combined_img = np.concatenate((img_scaled, pred_mask_scaled), axis=1)

                # Append combined image to video
                writer.append_data(combined_img)

        # Display the combined video
        st.subheader('Combined Images and Masks MP4 Video')
        video_file_combined = open(combined_video_path, 'rb')
        video_bytes_combined = video_file_combined.read()
        st.video(video_bytes_combined)
        col5, col6, col7 = st.columns(3)
        # Add download button for combined video
        col6.download_button(
            label="Download Combined Images and Masks MP4",
            data=video_bytes_combined,
            file_name='combined_video.mp4',
            mime='video/mp4'
        )
        video_file_combined.close()

def about():
    st.write("<div style='text-align: center'><h1>CELL SEGMENTATION AND TRACKING</h1></div>", unsafe_allow_html=True)
    st.write("Cell Segmentation and Tracking Using Deep Learning U-net, harnesses the capabilities of convolutional neural networks (CNNs) and the specialized U-net architecture for biomedical image analysis. By leveraging large datasets of labeled cell images, U-net accurately segments individual cells from complex backgrounds, overcoming challenges such as overlapping cells and varying intensities. Integration of tracking algorithms enables the monitoring of cell behavior over time, facilitating the analysis of dynamic processes like cell migration and proliferation. This framework accelerates discoveries in cellular biology and offers insights for medical research and clinical applications.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("cell-segmentation/test2.png", use_column_width=True)
    with col2:
        st.image("frame_1.png", use_column_width=True)
    



def trackpy():
    @pims.pipeline
    def gray(image):
        return np.array(image)[:, :, 1]  # Convert image to numpy array and take just the green channel
    st.subheader("Trackpy")
    uploaded_imgs = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_imgs is not None:
        frames = []
        for uploaded_img in uploaded_imgs:
            image = Image.open(io.BytesIO(uploaded_img.read()))
            frames.append(gray(image))
        
        if frames:
            features = tp.locate(frames[0], 11, invert=True)  # 11 - size of the features(needs to be an odd number) 
            features = tp.locate(frames[0], 11, invert=True, minmass=20)
            features = tp.batch(frames[:300], 11, minmass=20, invert=True)   
            t = tp.link(features, 5, memory=3)
            t1 = tp.filter_stubs(t, 25) 
            tp.mass_size(t1.groupby('particle').mean()) # convenience function -- just plots size vs. mass
            
            t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) &
                    (t1['ecc'] < 0.3))]
            plt.figure()
            tp.annotate(t2[t2['frame'] == 0], frames[0])
        else:
            st.write("")
     
def app():
    st.set_page_config(page_title="Cell Segmentation and Tracking", layout="wide")
    with st.sidebar:
        selection = sac.menu(
            items=[
                sac.MenuItem(label='Segmentation', type='group', children=[
                    sac.MenuItem(label='About'),
                    sac.MenuItem(label='Single Image Segmentation'),
                    sac.MenuItem(label='Multiple Image Segmentation'),
                ]),
                sac.MenuItem(label='Tracking', type='group', children=[
                    sac.MenuItem(label='Trackpy'),
                    sac.MenuItem(label='YOLOv8'),
                ]),
            ],
            key='About',
            open_all=True,
            indent=20,
            format_func='title',
            index=1

        )

    if selection == 'About':
        about()
    elif selection == 'Single Image Segmentation':
        images()
    elif selection == 'Multiple Image Segmentation':
        video()
    elif selection == 'Trackpy':
        trackpy()
    elif selection == 'YOLOv8':
        st.write("YOLOv8")

def main():
    app()

if __name__ == "__main__":
    main()

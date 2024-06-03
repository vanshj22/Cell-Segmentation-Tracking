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
import matplotlib.animation as animation

from skimage.io import imread, imshow
from skimage.transform import resize
from scipy import ndimage
from skimage import measure
from keras.models import load_model
import tifffile as tiff
import cv2
from PIL import Image, ImageDraw

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

# def images():
#     # Load the saved model
#     model = load_model(model_path)

#     # Display file uploader for test images
#     st.title('Upload Test Images For Single Image Segmentation')
#     uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

#     if uploaded_files:
#         pred_masks = []
#         uploaded_images = []
        
#         # Create two columns for displaying uploaded images and predicted masks
#         col3, col4 = st.columns(2)
        
#         for uploaded_file in uploaded_files:
#             # Read the uploaded image
#             image = Image.open(uploaded_file)
#             uploaded_images.append(image)
#             # Preprocess the image
#             test_image = preprocess_test_image(image)
            
#             # Expand the dimensions of the test image to create a batch of size 1
#             test_image_batch = np.expand_dims(test_image, axis=0)

#             # Predict segmentation mask
#             pred_mask = model.predict(test_image_batch)

#             # Threshold the predicted mask
#             pred_mask_thresholded = (pred_mask > 0.5).astype(np.uint8)
            
#             # Resize the predicted mask to match the height of the uploaded image
#             pred_mask_resized = np.array(Image.fromarray(pred_mask_thresholded.squeeze()).resize(image.size))
            
#             # Append the predicted mask to the list
#             pred_masks.append(pred_mask_resized)
            
#             # Display the uploaded image and predicted mask in the respective columns
#             col3.image(image, caption='Uploaded Image', use_column_width=True)
#             col4.image((pred_mask_resized * 255).astype(np.uint8), caption='Predicted Mask', use_column_width=True)

def std():
    with st.container():
        opt = st.sidebar.radio("Select the model", ["Predicted Mask", "Annotate Cells", "Plot Trajectory", "Plot Drift"])

        if opt == "Predicted Mask":
            # st.spinner(text="In progress...")

            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            IMG_CHANNELS = 3

            # Path to the saved model
            model_path = "cell-segmentation/U-net-model.h5"

            # Function to preprocess test images
            def preprocess_test_image(image):
                image = np.array(image)[:,:,:IMG_CHANNELS]
                image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                image = np.expand_dims(image, axis=-1)
                return image

            # Load the saved model
            model = load_model(model_path)

            st.title("Cell Segmentation with U-Net")

            uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

            if uploaded_files:
                if any(file.type == "image/tiff" for file in uploaded_files):
                    for file in uploaded_files:
                        if file.type == "image/tiff":
                            # Read the TIFF file
                            images = tiff.imread(file)

                            # Preprocess images
                            images = np.array([preprocess_test_image(img) for img in images])

                            # Predict masks
                            predicted_masks = model.predict(images, batch_size=1, verbose=1)

                            # Convert predictions to binary masks
                            binary_masks = (predicted_masks > 0.7).astype(np.uint8)

                            # Set up a plot to display images and masks side by side
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                            def update(frame):
                                for ax in axes:
                                    ax.clear()
                                # Display original image
                                axes[0].imshow(images[frame, :, :, 0], cmap='gray')
                                axes[0].set_title('Original Image')
                                axes[0].axis('off')
                                # Display predicted mask
                                axes[1].imshow(binary_masks[frame, :, :, 0], cmap='gray')
                                axes[1].set_title('Predicted Mask')
                                axes[1].axis('off')
                                fig.suptitle(f'Frame {frame + 1}/{len(images)}')

                            # Create animation
                            ani = animation.FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)

                            # Save the animation as a GIF
                            gif_path = 'segmentation_animation_unet.gif'
                            ani.save(gif_path, writer='imagemagick')

                            # Display the animation in Streamlit
                            st.image(gif_path, caption='Segmentation Animation', use_column_width=True)
                            break  # Only process the first TIFF file
                else:
                    images = [preprocess_test_image(Image.open(file)) for file in uploaded_files]
                    images = np.array(images)

                    # Predict masks
                    predicted_masks = model.predict(images, batch_size=1, verbose=1)

                    # Convert predictions to binary masks
                    binary_masks = (predicted_masks > 0.7).astype(np.uint8)

                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    def update(frame):
                        for ax in axes:
                            ax.clear()
                        # Display original image
                        axes[0].imshow(images[frame, :, :, 0], cmap='gray')
                        axes[0].set_title('Original Image')
                        axes[0].axis('off')
                        # Display predicted mask
                        axes[1].imshow(binary_masks[frame, :, :, 0], cmap='gray')
                        axes[1].set_title('Predicted Mask')
                        axes[1].axis('off')
                        fig.suptitle(f'Frame {frame + 1}/{len(images)}')

                    # Create animation
                    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)

                    # Save the animation as a GIF
                    gif_path = 'segmentation_animation_unet.gif'
                    ani.save(gif_path, writer='imagemagick')

                    st.image(gif_path, caption="Segmentation Animation", use_column_width=True)

        elif opt == "Annotate Cells":
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            IMG_CHANNELS = 3

            # Path to the saved model
            model_path = "cell-segmentation/U-net-model.h5"

            # Function to preprocess test images
            def preprocess_test_image(image):
                image = np.array(image)[:,:,:IMG_CHANNELS]
                image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                return image

            # Load the saved model
            model = load_model(model_path)

            # Streamlit app
            st.title("Cell Segmentation with U-Net")

            uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

            if uploaded_files:
                if any(file.type == "image/tiff" for file in uploaded_files):
                    for file in uploaded_files:
                        if file.type == "image/tiff":
                            # Read the TIFF file
                            images = tiff.imread(file)

                            # Preprocess images
                            images = np.array([preprocess_test_image(img) for img in images])

                            # Predict masks
                            predicted_masks = model.predict(images, batch_size=1, verbose=1)

                            # Convert predictions to binary masks
                            binary_masks = (predicted_masks > 0.7).astype(np.uint8)

                            # Annotate the images with the predicted masks
                            annotated_images = []
                            for i in range(len(images)):
                                # Convert to RGB
                                annotated_image = Image.fromarray((images[i, :, :, 0] * 255).astype(np.uint8)).convert('RGB')
                                draw = ImageDraw.Draw(annotated_image)
                                contours, _ = cv2.findContours(binary_masks[i, :, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour.shape) == 3:
                                        contour = contour.squeeze(axis=1)
                                    contour_points = [tuple(point) for point in contour]
                                    draw.line(contour_points + [contour_points[0]], fill=(255, 0, 0), width=1)  # Closing the contour
                                annotated_images.append(annotated_image)

                            # Create GIF
                            gif_path = 'segmentation_animation_unet.gif'
                            annotated_images[0].save(gif_path, save_all=True, append_images=annotated_images[1:], loop=0, duration=500)

                            # Display the animation in Streamlit
                            st.image(gif_path, caption='Segmentation Animation', use_column_width=True)
                            break  # Only process the first TIFF file
                else:
                    images = [preprocess_test_image(Image.open(file)) for file in uploaded_files]
                    images = np.array(images)

                    # Predict masks
                    predicted_masks = model.predict(images, batch_size=1, verbose=1)

                    # Convert predictions to binary masks
                    binary_masks = (predicted_masks > 0.7).astype(np.uint8)

                    # Annotate the images with the predicted masks
                    annotated_images = []
                    for i in range(len(images)):
                        # Convert to RGB
                        annotated_image = Image.fromarray((images[i, :, :, 0] * 255).astype(np.uint8)).convert('RGB')
                        draw = ImageDraw.Draw(annotated_image)
                        contours, _ = cv2.findContours(binary_masks[i, :, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour.shape) == 3:
                                contour = contour.squeeze(axis=1)
                            contour_points = [tuple(point) for point in contour]
                            draw.line(contour_points + [contour_points[0]], fill=(255, 0, 0), width=1)  # Closing the contour
                        annotated_images.append(annotated_image)

                    # Create GIF
                    gif_path = 'segmentation_animation_unet.gif'
                    annotated_images[0].save(gif_path, save_all=True, append_images=annotated_images[1:], loop=0, duration=500)

                    # Display the animation in Streamlit
                    st.image(gif_path, caption="Segmentation Animation", use_column_width=True)
def scinet():

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    # Path to the saved model
    model_path = "cell-segmentation/U-net-model.h5"

    # Function to preprocess test images
    def preprocess_test_image(image):
        image = np.array(image)[:,:,:IMG_CHANNELS]
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image = np.expand_dims(image, axis=-1)
        return image

    # Load the saved model
    model = load_model(model_path)

    st.title("Cell Segmentation with U-Net")

    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tif"])

    if uploaded_files:
        if any(file.type == "image/tiff" for file in uploaded_files):
            for file in uploaded_files:
                if file.type == "image/tiff":
                    # Read the TIFF file
                    images = tiff.imread(file)

                    # Preprocess images
                    images = np.array([preprocess_test_image(img) for img in images])

                    # Predict masks
                    predicted_masks = model.predict(images, batch_size=1, verbose=1)

                    # Convert predictions to binary masks
                    binary_masks = (predicted_masks > 0.7).astype(np.uint8)

                    # Set up a plot to display images and masks side by side
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    def update(frame):
                        for ax in axes:
                            ax.clear()
                        # Display original image
                        axes[0].imshow(images[frame, :, :, 0], cmap='gray')
                        axes[0].set_title('Original Image')
                        axes[0].axis('off')
                        # Display predicted mask
                        axes[1].imshow(binary_masks[frame, :, :, 0], cmap='gray')
                        axes[1].set_title('Predicted Mask')
                        axes[1].axis('off')
                        fig.suptitle(f'Frame {frame + 1}/{len(images)}')

                    # Create animation
                    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)

                    # Save the animation as a GIF
                    gif_path = 'segmentation_animation_unet.gif'
                    ani.save(gif_path, writer='imagemagick')

                    # Display the animation in Streamlit
                    st.image(gif_path, caption='Segmentation Animation', use_column_width=True)
                    break  # Only process the first TIFF file
        else:
            images = [preprocess_test_image(Image.open(file)) for file in uploaded_files]
            images = np.array(images)

            # Predict masks
            predicted_masks = model.predict(images, batch_size=1, verbose=1)

            # Convert predictions to binary masks
            binary_masks = (predicted_masks > 0.7).astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            def update(frame):
                for ax in axes:
                    ax.clear()
                # Display original image
                axes[0].imshow(images[frame, :, :, 0], cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                # Display predicted mask
                axes[1].imshow(binary_masks[frame, :, :, 0], cmap='gray')
                axes[1].set_title('Predicted Mask')
                axes[1].axis('off')
                fig.suptitle(f'Frame {frame + 1}/{len(images)}')

            # Create animation
            ani = animation.FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)

            # Save the animation as a GIF
            gif_path = 'segmentation_animation_unet.gif'
            ani.save(gif_path, writer='imagemagick')

            st.image(gif_path, caption="Segmentation Animation", use_column_width=True)



def images():   
    pass 

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
        

        # Create an MP4 video combining uploaded images and predicted masks
        combined_video_path = 'combined_video.mp4'
        writer = imageio.get_writer(combined_video_path, fps=1)

        try:
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

        finally:
            writer.close()


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
    

@pims.pipeline
def gray(image):
    return np.array(image)[:, :, 1] 


def trackpy():
    st.subheader("Trackpy")
    uploaded_imgs = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_imgs is not None:
        frames = []
        for uploaded_img in uploaded_imgs:
            # Open the uploaded image
            img = pims.ImageReader(uploaded_img)
            # Convert to grayscale
            gray_frame = gray(img)
            # Append to frames list
            frames.append(gray_frame)
            # Display the first frame
            st.image(gray_frame[0], caption='Uploaded Image', use_column_width=True)
        
        if frames:
            features = tp.locate(frames[0], 11, invert=True)  # 11 - size of the features(needs to be an odd number) 
            
            features = tp.annotate(features, frames[0])

            st.image(features, caption='Features', use_column_width=True)
            features = tp.batch(frames[:300], 11, minmass=20, invert=True)   
            t = tp.link(features, 5, memory=3)
            t1 = tp.filter_stubs(t, 25) 
            tp.mass_size(t1.groupby('particle').mean()) # convenience function -- just plots size vs. mass
            
            t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) &
                    (t1['ecc'] < 0.3))]
            plt.figure()
            # Determine if the image is RGB or grayscale
            if len(frames[0].shape) == 3 and frames[0].shape[2] in [3, 4]:  # RGB image
                tp.annotate(t2[t2['frame'] == 0], frames[0][:, :, 0])  # Assuming the first channel represents grayscale
            else:  # Grayscale image
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
                    sac.MenuItem(label='StarDist2D'),
                    sac.MenuItem(label='SCI-Net'),
                ])
            ],
            key='About',
            open_all=True,
            indent=20,
            format_func='title',
            index=1

        )

    if selection == 'About':
        about()
    elif selection == 'StarDist2D':
        std()
    elif selection == 'SCI-Net':
        scinet()
 

def main():
    app()

if __name__ == "__main__":
    main()

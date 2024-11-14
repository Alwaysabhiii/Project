import streamlit as st
import os
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Function to reduce the image using PCA
def reduce_image(file_name, accuracy):
    # Load and convert image to grayscale
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=accuracy)
    transformed_image = pca.fit_transform(gray_image)

    # Reconstruct the compressed image
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and save the compressed image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    output_path = 'Reduced_image.jpg'
    io.imsave(output_path, compressed_image_uint8)
    return output_path

# Streamlit UI
st.title("Image Compression using PCA")
st.write("Upload an image and select the accuracy for compression.")

# File upload input
uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Confidence/accuracy selection
accuracy = st.radio(
    "Select the compression accuracy:",
    options=[0.8, 0.9, 0.95, 0.99],
    index=1
)

if uploaded_image is not None:
    # Save the uploaded image
    image_path = os.path.join("uploads", uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Reduce the image using the selected accuracy
    st.write("Processing the image...")
    output_path = reduce_image(image_path, accuracy)

    # Display the compressed image
    st.image(output_path, caption="Compressed Image", use_column_width=True)

    # Download link for the reduced image
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Download Compressed Image",
            data=file,
            file_name="Reduced_image.jpg",
            mime="image/jpeg"
        )

from flask import Flask, request, render_template, send_file
import os
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    confidence = request.form.get('accuracy')

    if image.filename == '':
        return "No file selected", 400

    # Save the uploaded image
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Convert accuracy value to float and reduce the image
        accuracy = float(confidence)
        reduce_image(image_path, accuracy)

        # Render the index page with the download option
        return render_template('index.html', download_available=True)

    return "Upload failed", 500

@app.route('/download')
def download():
    file_path = 'Reduced_image.jpg'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

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
    io.imsave('Reduced_image.jpg', compressed_image_uint8)
    print("Image has been successfully compressed")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True,host='0.0.0.0',port=5050)

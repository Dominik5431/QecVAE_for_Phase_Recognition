from PIL import Image
import numpy as np

# Load the image
image_path = 'Ising_lattice.png'  # Replace with your image path
img = Image.open(image_path)

# Convert the image to grayscale for easier pixel analysis
img_gray = img.convert('L')
pixel_values = np.array(img_gray)

# Normalize pixel values between 0 and 1 for easier thresholding (0: black, 255: yellow)
normalized_pixels = pixel_values / 255.0

# Define thresholds for classification: black (close to 0) as +1, yellow (close to 1) as -1
threshold = 0.5
processed_pixels = np.where(normalized_pixels < threshold, 1, -1)

# Resize the processed pixel data to a 29x29 grid using nearest neighbor method
lattice_size = (29, 29)
lattice_image = Image.fromarray((processed_pixels * 127.5 + 127.5).astype(np.uint8)).resize(lattice_size, Image.NEAREST)

# Convert the resized image back to a numpy array and map back to +1 and -1
lattice_array = np.where(np.array(lattice_image) < 128, 1, -1)

# Print or inspect the 29x29 lattice array
print(lattice_array)
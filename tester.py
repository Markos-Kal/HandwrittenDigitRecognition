from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import interpolation
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")

def load_and_prepare_image(path):
	# Load image and convert to grayscale
	img = Image.open(path).convert('L')  # 'L' = grayscale mode

	# Invert image so background is black (0), digit is white (255)
	img = ImageOps.invert(img)

	# Convert to NumPy array
	arr = np.array(img)

	# Threshold: everything below 20 becomes 0 (background), helps with noise
	arr[arr < 20] = 0

	# Crop to the bounding box of the digit
	def crop_image(img_arr):
		coords = np.argwhere(img_arr > 0)
		if coords.shape[0] == 0:
			return np.zeros((28, 28))  # blank image if nothing found
		y0, x0 = coords.min(axis=0)
		y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
		cropped = img_arr[y0:y1, x0:x1]
		return cropped

	cropped = crop_image(arr)

	# Resize the digit to fit in 20x20 box
	img = Image.fromarray(cropped).resize((20, 20), Image.LANCZOS)
	resized = np.array(img)

	# Pad to 28x28 and center the digit
	def center_image(img_arr):
		# Pad with zeros to get 28x28
		rows, cols = img_arr.shape
		padded = np.pad(img_arr, (
			((28 - rows) // 2, (28 - rows + 1) // 2),
			((28 - cols) // 2, (28 - cols + 1) // 2)
		), 'constant')

		# Shift the image to center mass
		def shift_to_center(img):
			cy, cx = ndimage.center_of_mass(img)
			shift_y = int(np.round(14 - cy))
			shift_x = int(np.round(14 - cx))
			return interpolation.shift(img, shift=[shift_y, shift_x], mode='constant')

		from scipy import ndimage
		centered = shift_to_center(padded)
		return centered

	final = center_image(resized)

	# Show the processed image
	plt.imshow(final, cmap='gray')
	plt.title("Processed 28x28 Input")
	plt.axis('off')
	plt.show()


	return final.reshape(1, 28, 28, 1)  # or .reshape(1, 784) depending on model\


# === Inference Example ===
prepared_image = load_and_prepare_image("img.png")

# Optional: Save to CSV (flattened)
np.savetxt("test.csv", prepared_image.reshape(1, -1), delimiter=",", fmt="%.4f")

# Predict
predictions = model.predict(prepared_image)
predicted_label = np.argmax(predictions)

# Display
plt.imshow(prepared_image[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
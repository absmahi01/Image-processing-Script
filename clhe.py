import os
import cv2
import matplotlib.pyplot as plt

# Input and output folders
input_folder = "G:\\Mahi\\Pneumonia roboflow Gaublur\\train\\NORMAL"
output_folder = "G:\\Mahi\\brain_mri\\Pneumonia gaublur CLAHE roboflow\\train\\NORMAL"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 255])
        
        # Save grayscale image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_gray)
        
        # Plot and save histogram
        plt.figure()
        plt.plot(hist)
        plt.title('Histogram for ' + filename)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_folder, filename.split('.')[0] + '_histogram.png'))
        plt.close()

print("Processing completed.")

import os
import cv2

# Input and output directories
input_dir = 'G:/Mahi/brain_mri/lung and colon dataset/train/'
output_dir = 'G:/Mahi/brain_mri/brain_train_7222'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
        try:
            # Load the image
            input_image_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_image_path)
            
            if image is None:
                print(f"Failed to load image: {input_image_path}")
                continue

            # Apply Gaussian blur with kernel size 5x5 and standard deviation 0 (auto-calculated)
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

            # Save the processed image to the output directory
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, blurred_image)

            print(f"Processed image saved: {output_image_path}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

print("Processing complete.")
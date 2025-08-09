import os
from PIL import Image
import pillow_avif  # required for AVIF support

# Set input and output folder paths
input_folder = 'assets/circuits_avif'     # folder containing .avif files
output_folder = 'assets/circuits'         # folder to save .png files

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".avif"):
        avif_path = os.path.join(input_folder, filename)
        png_filename = filename.replace(".avif", ".png")
        png_path = os.path.join(output_folder, png_filename)

        try:
            # Open and convert the image
            with Image.open(avif_path) as img:
                img.save(png_path, "PNG")
            print(f"✅ Converted: {filename} → {png_filename}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
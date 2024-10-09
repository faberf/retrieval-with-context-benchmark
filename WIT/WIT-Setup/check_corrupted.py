import os
import glob
import json
from PIL import Image
import piexif

# Function to check if EXIF data is corrupted
def check_exif_corruption(image_path):
    try:
        img = Image.open(image_path)

        # Check if the image has EXIF data
        if img.info.get("exif") is None:
            print(f"No EXIF data found for {image_path}")
            return True

        # Load existing EXIF data
        exif_dict = piexif.load(img.info["exif"])

        # Check if user comment is present
        user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, None)
        if not user_comment:
            print(f"No user comment found in EXIF data for {image_path}")
            return True

        # Ensure user comment can be decoded as JSON
        try:
            user_comment = user_comment.decode('utf-8', errors='ignore')
            json.loads(user_comment)
        except Exception as e:
            print(f"Error decoding or parsing user comment in {image_path}: {e}")
            return True

        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return True

# Function to process images and identify those with corrupted EXIF data
def process_images(directory, output_file):
    image_files = glob.glob(os.path.join(directory, "*.*"))
    corrupted_images = []

    for image_path in image_files:
        if check_exif_corruption(image_path):
            corrupted_images.append(image_path)

    # Write the list of corrupted images to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corrupted_images, f, ensure_ascii=False, indent=4)

    print(f"Corrupted images have been saved to {output_file}")

# Specify the directory containing the images and output file name
image_directory = "../../benchmark/completed_images_with_categories"
output_file = "corrupted_images.json"

# Process the images and identify those with corrupted EXIF data
process_images(image_directory, output_file)

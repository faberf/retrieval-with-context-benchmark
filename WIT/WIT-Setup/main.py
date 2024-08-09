import argparse
import json
import time

import tensorflow as tf
import pandas as pd
import urllib.request
from PIL import Image
import piexif
import uuid
import logging

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a formatter to define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Create a file handler to write logs to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # You can set the desired log level for console output
console_handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

DEBUG = False

MIME_TYPES = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/bmp': 'bmp',
    'image/webp': 'webp',
    'image/tiff': 'tiff',
}

Image.MAX_IMAGE_PIXELS = 933120000

def main(args=None):
    filename = args.data_raw_path + 'wit_v1.train.all-1percent_sample.tsv'
    filename_out = args.datap_path + 'wit_v1.train.all-1percent_sample.csv'
    logger.info(f'Start reading file: {filename}')
    data = pd.read_csv(filename, sep='\t')
    logger.debug(f'File read successfully: {data.shape}')
    data.insert(0, 'image_path', None)
    download_images(data, args.images_root_path, filename_out)

    pass



def add_exif_metadata(image_path, json_object):
    try:
        # Convert JSON object to string
        json_str = json.dumps(json_object)

        # Open the image

        img = Image.open(image_path)

        if img.info.get("exif") is None:
            img.info["exif"] = piexif.dump({})


        # Load existing EXIF data
        exif_dict = piexif.load(img.info["exif"])

        exif_dict['Exif'][piexif.ExifIFD.UserComment] = json_str.encode('utf-8')

        # Insert the modified EXIF data back into the image
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, exif=exif_bytes)

        logger.info(f"JSON data added to {image_path}")
    except Exception as e:
        logger.warning(f'Error adding JSON data to {image_path}')


def download_images(data: pd.DataFrame, images_root_path, filename_out):
    for idx, row in data.iterrows():
        if DEBUG and idx > 10:
            break
        image_url = row['image_url']
        try:
            mime = MIME_TYPES[row['mime_type']]
        except:
            logger.warning(f'Error reading mime_type {row["mime_type"]} for image {idx}')
            continue

        image_path = images_root_path + str(uuid.uuid3(uuid.NAMESPACE_DNS, image_url)) + '.' + mime
        if download_image(image_url, image_path):
            logger.debug(f'Image {idx} downloaded successfully to {image_path}')
            data.at[idx, 'image_path'] = image_path
            row['image_path'] = image_path
            add_exif_metadata(image_path, row.to_dict())
            if idx % 100 == 0:
                logger.info(f'Processed {idx} images')
                data.to_csv(args.data_root_path + 'wit_v1.train.all-1percent_sample_with_image_path.csv', encoding='utf-8-sig')
        else:
            data.drop(idx, inplace=True)
    pass

def download_image(image_url, image_path) -> bool:
    try:
        success = retry_download_image(image_url, image_path)
        # Opening the image and displaying it (to confirm its presence)
        if DEBUG:
            img = Image.open(image_path)
            img.show()
        return success
    except Exception as e:
        logger.warning(f'Error downloading image: {image_path} from {image_url}')
        return False
    pass

def retry_download_image(image_url, image_path) -> bool:
    for i in range(5):
        try:
            urllib.request.urlretrieve(image_url, image_path)
            return True
        except Exception as e:
            logger.debug(f'Error downloading image: {image_path} from {image_url} - retrying {i + 1}')
            time.sleep(0.5+i)
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--data-raw-path',
                        default=r'/mnt/nas/data-raw/')
    parser.add_argument('-i', '--images-path',
                        default=r'/mnt/nas/images/')
    parser.add_argument('-d', '--data-path',
                        default=r'/mnt/nas/data/')

    args = parser.parse_args()
    main(args)

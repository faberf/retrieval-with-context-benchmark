import argparse
import json
import time

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
    filename_out = args.data_path + 'wit_v1.train.all-1percent_sample.csv'
    logger.info(f'Start reading file: {filename}')
    data = pd.read_csv(filename, sep='\t')
    logger.debug(f'File read successfully: {data.shape}')
    data.insert(0, 'image_path', None)
    download_images(data, args.images_path, filename_out)

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


import requests
from urllib.parse import urlparse, unquote

def get_wikipedia_language_and_title_from_url(url):
    """
    Extracts the Wikipedia language code and article title from a given Wikipedia URL.
    """
    parsed_url = urlparse(url)
    language_code = parsed_url.netloc.split('.')[0]
    title = parsed_url.path.split('/')[-1]
    return language_code, unquote(title)

def get_wikipedia_categories(language_code, title):
    """
    Fetches the categories of a given Wikipedia article using the Wikipedia API in the appropriate language.
    """
    api_url = f"https://{language_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "titles": title,
        "cllimit": "max"
    }
    
    response = requests.get(api_url, params=params)
    data = response.json()

    # Extract categories from the API response
    pages = data.get('query', {}).get('pages', {})
    for page_id, page_info in pages.items():
        if 'categories' in page_info:
            categories = [category['title'] for category in page_info['categories']]
            return categories
    
    return []

import random

def download_images(data: pd.DataFrame, images_root_path, filename_out):
    for idx, row in data.iterrows():
        if DEBUG and idx > 10:
            break
        image_url = row['image_url']
        language, title = get_wikipedia_language_and_title_from_url(row["page_url"])
        categories=get_wikipedia_categories(language, title)
        # select random category
        random_category = random.sample(categories, min(1, len(categories)))
        
        # select two random categories
        two_random_categories = random.sample(categories, min(2, len(categories)))
        
        # select five random categories
        five_random_categories = random.sample(categories, min(5, len(categories)))
        
        data.at[idx, 'categories'] = categories
        data.at[idx, 'random_category'] = random_category
        data.at[idx, 'two_random_categories'] = two_random_categories
        data.at[idx, 'five_random_categories'] = five_random_categories
        
        
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
                data.to_csv(args.data_raw_path + 'wit_v1.train.all-1percent_sample_with_image_path.csv', encoding='utf-8-sig')
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

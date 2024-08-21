import os
import glob
import json
from PIL import Image
import piexif
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from io import BytesIO
import base64

# Function to extract EXIF data, including the user comment, from an image using piexif
def extract_exif(image_path):
    try:
        img = Image.open(image_path)
        
        # Check if the image has EXIF data
        if img.info.get("exif") is None:
            print(f"No EXIF data found for {image_path}")
            return None
        
        # Load existing EXIF data
        exif_dict = piexif.load(img.info["exif"])
        
        # Extract the user comment if available
        user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, None)
        
        if user_comment:
            # Decode the user comment to string
            user_comment = user_comment.decode('utf-8', errors='ignore')
        
        return user_comment
    except Exception as e:
        print(f"Error extracting EXIF data from {image_path}: {e}")
        return None

# Function to convert an image to a Base64 data URL for inclusion in the message
def image_to_data_url(image_path):
    with open(image_path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{data}"

# Function to interact with ChatGPT-4 via LangChain and generate queries
def generate_queries_via_langchain(metadata, image_data_url):
    chat = ChatOpenAI(model="gpt-4o")
    
    prompt_template = f"""The image has the following metadata: 
{metadata}
    
Generate some multilingual queries that an archivist might use to search for this image. The queries should be very diverse and relate to different information needs an archivist might have.

For each query also make the following classifications:
Query references historic period (true/false)
Query references a location (true/false)
Query references traditional or communal activities, customs, or practices (true/false)
Query is in a language that is also utilized in image metadata (true/false)
Query references content of image that is not knowable from the metadata (true/false)
Query references an individual human (true/false)

The output should be in json format with no additional commentary:

[ 
    {{
    "query": "...", 
    "historic_period": true, 
    "location": false, 
    "traditional_customs_practices": true, 
    "language_in_metadata": false, 
    "references_content": false,
    "references_individual": false
    }}
]
    """

    # Create the content for the HumanMessage, including both the text and the image URL
    content = [{"type": "text", "text": prompt_template}]
    if image_data_url is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    
    # Create the HumanMessage with both metadata and image
    human_message = HumanMessage(content)

    # Get the response from the chat model
    response = chat([human_message])
    
    return json.loads(response.content)

import uuid

# Function to process images in a directory and generate queries
def process_images(directory, output_file):
    image_files = glob.glob(os.path.join(directory, "*.*"))
    all_queries = []

    for image_path in image_files:
        print(f"Processing {image_path}...")
        
        # Extract EXIF user comment
        exif_comment = extract_exif(image_path)
        
        # If there's no EXIF user comment, skip the image
        if exif_comment is None:
            print(f"No EXIF user comment found for {image_path}. Skipping...")
            continue
        
        # Convert the image to a data URL
        image_data_url = image_to_data_url(image_path)
        
        # Generate queries using LangChain and ChatGPT-4
        queries = generate_queries_via_langchain(exif_comment, image_data_url)
        
        image_url = json.loads(exif_comment)["image_url"]
        for query in queries:
            query["image_url"] = image_url
            query["image_id"] = str(uuid.uuid3(uuid.NAMESPACE_DNS, image_url))
        
        # Append the queries to the list
        all_queries.extend(queries)
    
    # Save all queries to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=4)

    print(f"All queries have been saved to {output_file}")

# Specify the directory containing the images and output file name
image_directory = "../../benchmark/images"
output_file = "output_queries.json"

# Process the images and generate queries
process_images(image_directory, output_file)

import os
import glob
import json
import random
from PIL import Image
import piexif
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to extract EXIF data, including the user comment and language, from an image using piexif
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
            user_comment_json = json.loads(user_comment)
            return user_comment_json

        return None
    except Exception as e:
        print(f"Error extracting EXIF data from {image_path}: {e}")
        return None

# Function to sample language codes based on EXIF language attribute and frequency
def sample_language_codes(language, valid_languages: list, language_frequencies: list):
    if language in valid_languages:
        # Find the index of the language to be removed
        index_to_remove = valid_languages.index(language)
        # Remove the language and its corresponding frequency
        valid_languages = valid_languages[:index_to_remove] + valid_languages[index_to_remove + 1:]
        language_frequencies = language_frequencies[:index_to_remove] + language_frequencies[index_to_remove + 1:]
    
    # Sample with the updated valid_languages and language_frequencies
    return [language] * 4 + random.choices(valid_languages, weights=language_frequencies, k=6)


# Function to convert an image to a Base64 data URL for inclusion in the message
def image_to_data_url(image_path):
    with open(image_path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{data}"

import json

# Function to clean the OpenAI response content by removing anything before the first [ and after the last ]
def clean_response_content(response_content):
    try:
        # Find the position of the first [ and last ] to extract valid JSON
        start = response_content.find('[')
        end = response_content.rfind(']')
        
        # Ensure both brackets are found
        if start != -1 and end != -1:
            # Extract the valid JSON content
            cleaned_content = response_content[start:end+1]
            return cleaned_content
        else:
            raise ValueError("Response content does not contain valid JSON brackets.")
    except Exception as e:
        print(f"Error cleaning response content: {e}")
        return None

# Function to interact with ChatGPT-4 via LangChain and generate queries in specific languages
def generate_queries_via_langchain(metadata, image_data_url, languages):
    chat = ChatOpenAI(model="gpt-4o")

    prompt_template = f"""The image has the following metadata: 
{metadata}

Generate exactly 10 multilingual search queries for which this image would be a helpful result. The queries should be diverse and related to different information needs an archivist or researcher might have. Use the following languages: {', '.join(languages)}. Make sure to use a variety of keywords in the queries (avoid repeating the same location name or term in every query). The queries should resemble typical search engine queries, not questions.

For each query, also make the following classifications:
- Query references a specific historic period (true/false). Example: "Reconnaissance techniques WWII" (true), "Development of Central Park NYC" (false if no specific historical period is mentioned).
- Query references a specific location (true/false). Example: "Urban planning Berlin" (true), "Building changes over time" (false).
- Query references traditional or communal activities, customs, or practices (true/false). **Only mark this as true if the query directly relates to a specific event, ritual, festival, or communal tradition that is actively practiced or celebrated.**
   - **True examples**: "Harvest Festival rural communities," "Spring Equinox celebrations traditions."
   - **False examples**: "Cultural significance local storyteller" (references a cultural figure, not a specific custom), "Symbolism of animals in folklore" (symbolism is not a specific practice), "Traditional church architecture" (architecture is not a custom or practice), "Mythological significance regional folklore" (mythological references, not active customs).
- **Goes_beyond_metadata_using_image_contents** (true/false): Mark this as **true** if the query relies on information that requires visual inspection or interpretation of the image, which goes beyond what is provided in the metadata. Mark it **false** if the query can be fully answered by the metadata alone (e.g., image captions, descriptions).
   - **True examples**: "Statue Covered in Vines" (if the metadata doesn’t mention the statue’s condition or that it’s covered in vines), "Old Wooden Ship in Harbor" (if the metadata doesn’t describe the ship’s material or its location in a harbor), "1950s Advertisement Poster" (if the metadata explicitly describes it as a 1950s advertisement, but it’s not clear from the metadata that it is a poster).
   - **False examples**: "Flag of Amsterdam" (if the metadata explicitly states the image is of a flag), "Two Soldiers Standing in a Battlefield" (if the metadata also mentions the number of people as part of the battlefield context).
- Query references an individual human (true/false). Example: "John Smith architect" (true), "Building management" (false).

The output should be in JSON format with no additional commentary:
[ 
    {{
    "query": "...", 
    "historic_period": true, 
    "location": false, 
    "traditional_customs_practices": true, 
    "goes_beyond_metadata_using_image_contents": false,
    "references_individual": false
    }},
    ...
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

    try:
        # Clean the response to remove extraneous text and keep only valid JSON
        cleaned_response = clean_response_content(response.content)
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        return generate_queries_via_langchain(metadata, image_data_url, languages)
import math

from urllib.parse import urlparse



# Thread function to process a single image
def process_single_image(image_path, valid_languages, language_frequencies):
    # Extract EXIF user comment and language
    exif_data = extract_exif(image_path)

    # If there's no EXIF user comment, skip the image
    if exif_data is None or not exif_data.get("image_url"):
        print(f"No EXIF user comment or image URL found for {image_path}. Skipping...")
        return None

    # Get the language from EXIF data
    metadata_language = exif_data.get("language")  
    if metadata_language is None or not type(metadata_language) == str:
        url = urlparse(exif_data.get("image_url"))
        metadata_language = url.netloc.split('.')[0]
    

    # Convert the image to a data URL
    image_data_url = image_to_data_url(image_path)

    # Sample query languages based on the image's language
    query_languages = sample_language_codes(metadata_language, valid_languages, language_frequencies)

    # Generate queries using LangChain and ChatGPT-4
    queries = generate_queries_via_langchain(exif_data, image_data_url, query_languages)

    # Use the image filename (without extension) as the image_id
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # Assign the correct language from query_languages to each query, and add the new fields
    for idx, query in enumerate(queries):
        query["image_url"] = exif_data["image_url"]
        query["image_id"] = image_id
        query_language = query_languages[idx % len(query_languages)]  # Use query languages cyclically if necessary
        query["query_language"] = query_language
        query["metadata_language"] = metadata_language
        query["query_language_equals_metadata_language"] = (query_language == metadata_language)
        query["page_url"] = exif_data.get("page_url", "")
        query["image_file_name"] = os.path.basename(image_path)
        query["overcategory"] = exif_data.get("overcategory", "")

    return queries



# Function to process images with multithreading
def process_images(directory, output_file, checkpoint_file="processed_images_improved.json"):
    image_files = glob.glob(os.path.join(directory, "*.*"))

    # Load progress from checkpoint if it exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            processed_images = set(json.load(f))
    else:
        processed_images = set()

    # Filter out already processed images
    image_files = [img for img in image_files if img not in processed_images]

    # Open the output file and check its current state
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        # If the file doesn't exist or is empty, write the opening bracket
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
        needs_comma = False
    else:
        # If the file exists and has content, prepare to append
        needs_comma = True

    # Define valid languages and their frequencies (from the previous language frequency data)
    valid_languages = ['en', 'de', 'fr', 'it', 'es', 'ru', 'nl', 'ja', 'sv', 'uk', 'zh-TW', 'ca', 'pt', 'zh', 'cs', 'ar', 'vi', 'ro', 'fi', 'no', 'iw', 'sr', 'tr']
    language_frequencies = [422, 245, 190, 118, 115, 115, 101, 79, 76, 71, 67, 60, 54, 53, 53, 49, 45, 37, 34, 33, 33, 31, 31]

    # Create a ThreadPoolExecutor with 10 threads (equal to batch size)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(0, len(image_files), 10):
            # Get a batch of 10 images
            image_batch = image_files[i:i+10]

            # Submit all 10 images to the thread pool
            futures = {executor.submit(process_single_image, img, valid_languages, language_frequencies): img for img in image_batch}

            # Gather the results and write them once the batch is done
            batch_queries = []
            batch_processed_images = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch_queries.extend(result)
                    batch_processed_images.append(futures[future])

            # If there are queries from the batch, write them to the output file
            if batch_queries:
                with open(output_file, 'a', encoding='utf-8') as f:
                    if needs_comma:
                        f.write(',\n')  # Add a comma before the next JSON object if needed
                    f.write(",\n".join(json.dumps(q, ensure_ascii=False, indent=4) for q in batch_queries))
                    needs_comma = True  # Ensure subsequent entries are comma-separated

                # Update processed images and save checkpoint after each batch
                processed_images.update(batch_processed_images)
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(list(processed_images), f, ensure_ascii=False, indent=4)

    # After all images are processed, close the JSON array properly
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')  # Close the JSON array

    print(f"Progress saved. All queries have been saved to {output_file}")

# Specify the directory containing the images and output file name
image_directory = "../../benchmark/completed_images_with_categories"
output_file = "output_queries_improved.json"

# Process the images and generate queries
process_images(image_directory, output_file)

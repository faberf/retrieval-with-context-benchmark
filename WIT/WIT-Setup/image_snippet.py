import requests
import json
import time
import os
from PIL import Image, ImageDraw, ImageFont
import psycopg2

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64
from langchain_community.tools import DuckDuckGoSearchResults

# Database connection settings
DB_HOST = "10.34.64.130"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "admin"

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

conn.autocommit = False
cur = conn.cursor()

from abc import ABC, abstractmethod
import math

# Base Retriever class using abc
class Retriever(ABC):
    def __init__(self, schema_name, host, max_retries=1000, retry_delay=2, retrieval_limit=1000):
        self.schema_name = schema_name
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retrieval_limit = retrieval_limit

    @abstractmethod
    def make_payload(self, input_text):
        pass

    @abstractmethod
    def create_table_query(self):
        pass

    @abstractmethod
    def check_existing_result(self, query_id):
        pass

    @abstractmethod
    def add_result_to_db(self, query_id, ranking):
        pass

    def query_files(self, input_text):
        url = f"{self.host}/api/{self.schema_name}/query"
        payload = self.make_payload(input_text)
        headers = {'Content-Type': 'application/json'}
        retries = 0

        while retries < self.max_retries:
            try:
                response = requests.post(url, data=json.dumps(payload), headers=headers)
                response.raise_for_status()
                response_data = response.json()

                filenames = []
                if "retrievables" in response_data:
                    for item in response_data["retrievables"]:
                        path = item["properties"].get("path", "")
                        if path:
                            base_name = os.path.basename(path)
                            file_name, _ = os.path.splitext(base_name)
                            filenames.append(base_name)
                
                if filenames:
                    return filenames
                else:
                    print(f"Empty result from API, retrying... ({retries + 1}/{self.max_retries})")
                    retries += 1
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                retries += 1
                time.sleep(self.retry_delay)

        print(f"Failed to retrieve results after {self.max_retries} attempts.")
        return []

    @abstractmethod
    def get_method_name(self):
        pass

    @abstractmethod
    def get_table_name(self):
        pass

    @abstractmethod
    def get_rank_key(self):
        pass
    
    @abstractmethod
    def get_results(self, cur):
        pass

# ClipRetriever class
class ClipRetriever(Retriever):
    def make_payload(self, input_text):
        return {
            "inputs": {
                "mytext1": {"type": "TEXT", "data": input_text}
            },
            "operations": {
                "clip1": {"type": "RETRIEVER", "field": "clip", "input": "mytext1"},
                "filelookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "clip1"}
            },
            "context": {
                "global": {
                    "limit": str(self.retrieval_limit)
                },
                "local": {"filelookup": {"field": "file", "keys": "path"}}
            },
            "output": "filelookup"
        }

    def create_table_query(self):
        table_name = "clip_results"
        return f"""
        CREATE TABLE IF NOT EXISTS image_query_schema.{table_name} (
            query_id INT REFERENCES image_query_schema.query(query_id),
            ranking TEXT[],
            retrieval_schema TEXT NOT NULL,
            UNIQUE (query_id, retrieval_schema)
        );
        """

    def check_existing_result(self, query_id):
        table_name = "clip_results"
        cur.execute(f"SELECT query_id FROM image_query_schema.{table_name} WHERE query_id = %s AND retrieval_schema = %s", (query_id, self.schema_name))
        return cur.fetchone() is not None

    def add_result_to_db(self, query_id, ranking):
        table_name = "clip_results"
        cur.execute(f"""
            INSERT INTO image_query_schema.{table_name} (query_id, ranking, retrieval_schema)
            VALUES (%s, %s, %s)
            ON CONFLICT (query_id, retrieval_schema) DO UPDATE 
            SET ranking = EXCLUDED.ranking;
        """, (query_id, ranking, self.schema_name))

    def get_method_name(self):
        return f"CLIP ({self.schema_name})"

    def get_table_name(self):
        return "clip_results"

    def get_rank_key(self):
        return f"clip_rank_{self.schema_name}"
    
    def get_results(self, cur):
        """Fetch results as a list of lists of ranks for each query."""
        table_name = self.get_table_name()
        sql = f"""
            SELECT q.query_id, i.image_file_name, r.ranking
            FROM image_query_schema.{table_name} r
            JOIN image_query_schema.query q ON r.query_id = q.query_id
            JOIN image_query_schema.query_image qi ON q.query_id = qi.query_id
            JOIN image_query_schema.image i ON qi.image_id = i.image_id
            WHERE r.retrieval_schema = %s
        """
        params = [self.schema_name]
        cur.execute(sql, params)
        results = cur.fetchall()

        from collections import defaultdict
        ranks_by_query = defaultdict(list)

        for query_id, image_file_name, ranking in results:
            # Extract the base filename without extension
            image_id = os.path.splitext(image_file_name)[0]
            if image_id in ranking:
                rank = ranking.index(image_id) + 1
            else:
                rank = None
            ranks_by_query[query_id].append(rank)

        # Collect and sort ranks per query
        all_ranks = []
        for query_id in sorted(ranks_by_query.keys()):
            ranks = ranks_by_query[query_id]
            # Sort ranks from best to worst, placing None at the end
            sorted_ranks = sorted(
                ranks, key=lambda x: (x is None, x)
            )
            all_ranks.append(sorted_ranks)

        return all_ranks
    
class CaptionDenseRetriever(Retriever):
    def make_payload(self, input_text):
        return {
            "inputs": {
                "mytext1": {"type": "TEXT", "data": input_text}
            },
            "operations": {
                "captionDense1": {"type": "RETRIEVER", "field": "captiondense", "input": "mytext1"},
                "filelookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "captionDense1"}
            },
            "context": {
                "global": {
                    "limit": str(self.retrieval_limit)
                },
                "local": {"filelookup": {"field": "file", "keys": "path"}}
            },
            "output": "filelookup"
        }

    def create_table_query(self):
        table_name = "caption_dense_results"
        return f"""
        CREATE TABLE IF NOT EXISTS image_query_schema.{table_name} (
            query_id INT REFERENCES image_query_schema.query(query_id),
            ranking TEXT[],
            retrieval_schema TEXT NOT NULL,
            UNIQUE (query_id, retrieval_schema)
        );
        """

    def check_existing_result(self, query_id):
        table_name = "caption_dense_results"
        cur.execute(f"SELECT query_id FROM image_query_schema.{table_name} WHERE query_id = %s AND retrieval_schema = %s", (query_id, self.schema_name))
        return cur.fetchone() is not None

    def add_result_to_db(self, query_id, ranking):
        table_name = "caption_dense_results"
        cur.execute(f"""
            INSERT INTO image_query_schema.{table_name} (query_id, ranking, retrieval_schema)
            VALUES (%s, %s, %s)
            ON CONFLICT (query_id, retrieval_schema) DO UPDATE 
            SET ranking = EXCLUDED.ranking;
        """, (query_id, ranking, self.schema_name))

    def get_method_name(self):
        return f"CaptionDense ({self.schema_name})"

    def get_table_name(self):
        return "caption_dense_results"

    def get_rank_key(self):
        return f"captiondense_rank_{self.schema_name}"
    
    def get_results(self, cur):
        """Fetch results as a list of lists of ranks for each query."""
        table_name = self.get_table_name()
        sql = f"""
            SELECT q.query_id, i.image_file_name, r.ranking
            FROM image_query_schema.{table_name} r
            JOIN image_query_schema.query q ON r.query_id = q.query_id
            JOIN image_query_schema.query_image qi ON q.query_id = qi.query_id
            JOIN image_query_schema.image i ON qi.image_id = i.image_id
            WHERE r.retrieval_schema = %s
        """
        params = [self.schema_name]
        cur.execute(sql, params)
        results = cur.fetchall()

        from collections import defaultdict
        ranks_by_query = defaultdict(list)

        for query_id, image_file_name, ranking in results:
            # Extract the base filename without extension
            image_id = os.path.splitext(image_file_name)[0]
            if image_id in ranking:
                rank = ranking.index(image_id) + 1
            else:
                rank = None
            ranks_by_query[query_id].append(rank)

        # Collect and sort ranks per query
        all_ranks = []
        for query_id in sorted(ranks_by_query.keys()):
            ranks = ranks_by_query[query_id]
            # Sort ranks from best (smallest) to worst (largest), placing None at the end
            sorted_ranks = sorted(
                ranks, key=lambda x: (x is None, x)
            )
            all_ranks.append(sorted_ranks)

        return all_ranks


# ClipDenseCaptionFusionRetriever class
class ClipDenseCaptionFusionRetriever(Retriever):
    def __init__(self, schema_name, host, clip_weight=0.5, caption_dense_weight=0.5, p=1.0, max_retries=1000, retry_delay=2, retrieval_limit=1000):
        assert clip_weight + caption_dense_weight == 1.0, "Weights should sum to 1.0"
        self.clip_weight = clip_weight
        self.caption_dense_weight = caption_dense_weight
        self.p = p
        super().__init__(schema_name, host, max_retries, retry_delay, retrieval_limit)

    def make_payload(self, input_text):
        p_str = "Infinity" if math.isinf(self.p) else str(self.p)
        return {
            "inputs": {
                "query": {"type": "TEXT", "data": input_text}
            },
            "operations": {
                "feature1": {"type": "RETRIEVER", "field": "clip", "input": "query"},
                "feature2": {"type": "RETRIEVER", "field": "captiondense", "input": "query"},
                "score": {"type": "AGGREGATOR", "aggregatorName": "WeightedScoreFusion", "inputs": ["feature1", "feature2"]},
                "aggregator": {"type": "TRANSFORMER", "transformerName": "ScoreAggregator", "input": "score"},
                "filelookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
            },
            "context": {
                "global": {
                    "limit": str(self.retrieval_limit)
                },
                "local": {
                    "filelookup": {"field": "file", "keys": "path"},
                    "score": {"weights": f"{self.clip_weight},{self.caption_dense_weight}", "p": str(p_str)}
                }
            },
            "output": "filelookup"
        }

    def create_table_query(self):
        table_name = "clip_dense_caption_fusion_results"
        return f"""
        CREATE TABLE IF NOT EXISTS image_query_schema.{table_name} (
            query_id INT REFERENCES image_query_schema.query(query_id),
            ranking TEXT[],
            clip_weight FLOAT NOT NULL,
            caption_dense_weight FLOAT NOT NULL,
            p FLOAT NOT NULL,
            retrieval_schema TEXT NOT NULL,
            UNIQUE (query_id, clip_weight, caption_dense_weight, p, retrieval_schema)
        );
        """

    def check_existing_result(self, query_id):
        table_name = "clip_dense_caption_fusion_results"
        cur.execute(f"""
            SELECT query_id FROM image_query_schema.{table_name}
            WHERE query_id = %s AND retrieval_schema = %s
            AND clip_weight = %s AND caption_dense_weight = %s AND p = %s
        """, (query_id, self.schema_name, self.clip_weight, self.caption_dense_weight, self.p))
        return cur.fetchone() is not None

    def add_result_to_db(self, query_id, ranking):
        table_name = "clip_dense_caption_fusion_results"
        cur.execute(f"""
            INSERT INTO image_query_schema.{table_name} (query_id, ranking, clip_weight, caption_dense_weight, p, retrieval_schema)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (query_id, clip_weight, caption_dense_weight, p, retrieval_schema) DO UPDATE 
            SET ranking = EXCLUDED.ranking;
        """, (query_id, ranking, self.clip_weight, self.caption_dense_weight, self.p, self.schema_name))

    def get_method_name(self):
        return f"Fusion ({self.schema_name}, weights={self.clip_weight}/{self.caption_dense_weight}, p={self.p})"

    def get_table_name(self):
        return "clip_dense_caption_fusion_results"

    def get_rank_key(self):
        return f"fusion_rank_{self.schema_name}_{self.clip_weight}_{self.caption_dense_weight}_p{self.p}"

    def get_results(self, cur):
            """Fetch results as a list of lists of ranks for each query."""
            table_name = self.get_table_name()
            sql = f"""
                SELECT q.query_id, i.image_file_name, r.ranking
                FROM image_query_schema.{table_name} r
                JOIN image_query_schema.query q ON r.query_id = q.query_id
                JOIN image_query_schema.query_image qi ON q.query_id = qi.query_id
                JOIN image_query_schema.image i ON qi.image_id = i.image_id
                WHERE r.retrieval_schema = %s
                AND r.clip_weight = %s AND r.caption_dense_weight = %s AND r.p = %s
            """
            params = [self.schema_name, self.clip_weight, self.caption_dense_weight, self.p]
            cur.execute(sql, params)
            results = cur.fetchall()

            from collections import defaultdict
            ranks_by_query = defaultdict(list)

            for query_id, image_file_name, ranking in results:
                # Extract the base filename without extension
                image_id = os.path.splitext(image_file_name)[0]
                if image_id in ranking:
                    rank = ranking.index(image_id) + 1
                else:
                    rank = None
                ranks_by_query[query_id].append(rank)

            # Collect and sort ranks per query
            all_ranks = []
            for query_id in sorted(ranks_by_query.keys()):
                ranks = ranks_by_query[query_id]
                # Sort ranks from best to worst, placing None at the end
                sorted_ranks = sorted(
                    ranks, key=lambda x: (x is None, x)
                )
                all_ranks.append(sorted_ranks)

            return all_ranks

# Function to create the image grid
def create_image_grid(image_filenames, output_filename, padding=10, max_size=1000):
    if len(image_filenames) != 9:
        raise ValueError("Exactly 9 image filenames are required.")
    
    # Labels for each image
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    # Find the largest width and height among all images
    max_width, max_height = 0, 0
    for filename in image_filenames:
        with Image.open(filename) as img:
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
    
    # Use the smaller of max_size and the largest dimension as the target size
    target_size = min(max(max_width, max_height), max_size)
    
    # Load and process images
    processed_images = []
    for idx, filename in enumerate(image_filenames):
        img = Image.open(filename)
        
        # Determine the scaling factor based on the largest dimension
        if img.width > img.height:
            scale_factor = target_size / img.width
            new_size = (target_size, int(img.height * scale_factor))
        else:
            scale_factor = target_size / img.height
            new_size = (int(img.width * scale_factor), target_size)
        
        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create a new image with a white background and paste the resized image onto it
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        paste_position = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
        new_img.paste(img_resized, paste_position)
        
        # Calculate font size based on the target size (15% of the width of the subimage)
        font_size = int(target_size * 0.15)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw the label with a white border around it
        draw = ImageDraw.Draw(new_img)
        text_bbox = draw.textbbox((0, 0), labels[idx], font=font)
        label_position = (10, 10)

        # Draw the white border in a single pass using offsets
        border_thickness = int(0.06 * font_size) + 1
        offsets = [
            (x_offset, y_offset)
            for x_offset in range(-border_thickness, border_thickness + 1)
            for y_offset in range(-border_thickness, border_thickness + 1)
            if x_offset != 0 or y_offset != 0  # Avoid drawing at the center
        ]

        # Draw the border only once for each offset
        for offset in offsets:
            border_position = (label_position[0] + offset[0], label_position[1] + offset[1])
            draw.text(border_position, labels[idx], fill=(255, 255, 255), font=font)

        # Draw the black text on top of the border
        draw.text(label_position, labels[idx], fill=(0, 0, 0), font=font)
        
        processed_images.append(new_img)
    
    # Calculate grid dimensions with padding
    grid_width = target_size * 3 + padding * 2
    grid_height = target_size * 3 + padding * 2
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Paste each processed image into the grid with padding
    for i, image in enumerate(processed_images):
        x_offset = (i % 3) * (target_size + padding) + padding
        y_offset = (i // 3) * (target_size + padding) + padding
        grid_image.paste(image, (x_offset, y_offset))
    
    # Save the final grid image
    grid_image.save(output_filename)

import piexif

import ast
import re

def extract_dict_from_string(s):
    # Find all occurrences of '{' and '}'
    open_braces = [m.start() for m in re.finditer(r'{', s)]
    close_braces = [m.start() for m in re.finditer(r'}', s)]
    
    # Start with the last closing brace
    for close_pos in reversed(close_braces):
        for open_pos in reversed(open_braces):
            if open_pos < close_pos:
                dict_str = s[open_pos:close_pos + 1]
                try:
                    # Try to parse the potential dictionary string
                    return ast.literal_eval(dict_str)
                except (SyntaxError, ValueError):
                    # On error, skip to the next pair of braces
                    continue
    return None





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

def add_new_mappings_atomic(query_id, image_files, explanations):
    try:
        # Begin the transaction
        insert_query = """
        INSERT INTO image_query_schema.query_image (query_id, image_id, reason)
        VALUES (%s, %s, %s)
        ON CONFLICT (query_id, image_id) DO NOTHING;
        """
        
        # Execute the insert for each image_file
        for image_file, explanation in zip(image_files, explanations):
            image_id, _ = os.path.splitext(image_file)  # Extract image_id
            cur.execute(insert_query, (query_id, image_id, explanation))

        # Commit the transaction
        conn.commit()
        print(f"Inserted entries for query_id {query_id}")
    
    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        print(f"Transaction failed, rolled back: {e}")
        
def has_entries(query_id):
    # Check if the query_id has at least one entry in the query_image table
    check_query = """
    SELECT EXISTS (
        SELECT 1 FROM image_query_schema.query_image
        WHERE query_id = %s
    );
    """
    
    cur.execute(check_query, (query_id,))
    exists = cur.fetchone()[0]  # fetchone returns a tuple, the first element is the boolean result
    return exists

# Function to convert an image to a Base64 data URL for inclusion in the message
def image_to_data_url(image_path):
    with open(image_path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{data}"

# Fetch all queries and associated image filenames
cur.execute("""
    SELECT q.query_id, q.query_text, i.image_file_name
    FROM image_query_schema.query q
    JOIN image_query_schema.image i ON q.original_image_id = i.image_id;
""")
queries = cur.fetchall()

import json

# Initialize the retriever
host = "http://10.34.64.84:7071"
retriever = ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.6, caption_dense_weight=0.4)


def get_retrieved_paths(query, original_image_filename):
    # Query for filenames using the given query
    retrieved_paths = retriever.query_files(query)
    try:
        retrieved_paths.remove(original_image_filename)
    except ValueError:
        print("Original image not found in retrieved paths")
    
    return [[original_image_filename] + retrieved_paths[i:i+8] for i in range(0, len(retrieved_paths)-16, 8)]


def make_prompt(query):
    return (
        "Below is an image grid with images labeled A through I. "
        f"These images were retrieved for an archivist who searched for the query '{query}'. "
        f"For each image, please explain why it is or is not a correct result for the query '{query}'. "
        "Think carefully and consider every aspect of the query, the visible image content, and any metadata available. "
        "If information is missing or unclear, use web search to find additional context by carefully selecting a search term that helps you determine relevancy. "
        "Make sure not to include image labels such as 'Image X' in your search term, as these labels are only for your reference. "
        "Make sure to search the web for each image until you have all necessary information to make a decision. "
        "Keep in mind that web search results may not always be accurate, relevant or up-to-date and should thus be ignored. "
        "Be very strict in your evaluation and consider an image incorrect when in doubt. "
        "Err on the side of caution and only mark an image as correct if you are certain it is relevant to the query. "
        "When explaining why an image is correct or incorrect, provide as much detail as possible and include sources from the web if relevant. "
        f"As a reference, Image A is definitely a correct result for the query '{query}'. "
        "Finally, after thinking through step by step, provide a python dictionary with labels of images that are correct results and explanation strings as values. "
        f"Do not include labels of images that are incorrect results for the query '{query}' in the dictionary. "
    )
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
import time
last_time = time.time()

class MyTool(DuckDuckGoSearchResults):
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        global last_time
        while time.time() - last_time < 10:
            time.sleep(1)
        last_time = time.time()
        return super()._run(query, run_manager)

tools = [MyTool()]
        
# Send the generated image to the chat model
chat = ChatOpenAI(model="gpt-4o").bind_tools(tools)

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def send_to_chat_model(human_message):
    
    prompt = ChatPromptTemplate.from_messages([human_message, MessagesPlaceholder(variable_name="agent_scratchpad")])
    agent = (
    {
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | chat
    | OpenAIToolsAgentOutputParser()
)


    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    try:
        invocation = agent_executor.invoke({"human_message": human_message})
    except Exception as e:
        return send_to_chat_model(human_message)
    return invocation["output"]
    
def remove_image_and_capitalize(input_string):
    # Remove 'image' in any case format and strip extra spaces
    cleaned_string = re.sub(r'\bimage\b', '', input_string, flags=re.IGNORECASE).strip()
    
    # Check if there is more than one character left (e.g., "A and B")
    remaining_chars = cleaned_string.split()
    if len(remaining_chars) > 1:
        raise ValueError("More than one letter remains after removing 'image'.")

    # Return the remaining character in uppercase if exists
    return remaining_chars[0].upper() if remaining_chars else None

def process_query(query_id, query, original_image_filename, retrieved_paths_list):
    try:
        print(f"Processing Query {query_id}: {query}")

        # Initialize an outermost dictionary to accumulate results, using image paths as keys
        outer_extracted_dict = {}

        # Iterate over the list of lists of retrieved paths
        for batch_index, retrieved_paths in enumerate(retrieved_paths_list):
            # Prepend the correct path to the retrieved paths
            image_directory = '../../benchmark/completed_images_with_categories/'
            image_filenames = [os.path.join(image_directory, os.path.basename(path)) for path in retrieved_paths]

            # Output file name with unique identifier (include batch index)
            output_filename = f'image_grid_test_output_{query_id}_{batch_index}.jpg'

            # Check if we have enough images to create the grid
            if len(image_filenames) >= 9:
                # Extract metadata for each of the first 9 images
                image_metadata = []
                for image_path in image_filenames[:9]:
                    metadata = extract_exif(image_path)
                    image_metadata.append({"image_path": image_path, "metadata": metadata})

                # Create the image grid
                create_image_grid(image_filenames[:9], output_filename)

                # Generate the content for the HumanMessage
                prompt = make_prompt(query)
                content = [{"type": "text", "text": prompt}]

                # Add the labels and metadata details
                labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
                for label, meta in zip(labels, image_metadata):
                    meta_text = f"Image {label}: {meta['metadata']}" if meta['metadata'] else f"Image {label}: No metadata available"
                    content.append({"type": "text", "text": meta_text})

                # Convert the image to a data URL
                image_data_url = image_to_data_url(output_filename)

                if image_data_url:
                    content.append({"type": "image_url", "image_url": {"url": image_data_url}})

                # Create the HumanMessage with both metadata and image
                human_message = HumanMessage(content)

                # Get the response from the chat model
                response = send_to_chat_model(human_message)

                # Parse the response into a dictionary (label -> explanation)
                extracted_dict = extract_dict_from_string(response)

                # Delete the temporary output file after use
                if os.path.exists(output_filename):
                    os.remove(output_filename)

                # Map the retrieved image paths to the corresponding explanations
                path_explanation_mapping = {
                    retrieved_paths[labels.index(remove_image_and_capitalize(label))]: explanation
                    for label, explanation in extracted_dict.items()
                }
                
                if not retrieved_paths[0] in path_explanation_mapping:
                    path_explanation_mapping[retrieved_paths[0]] = "ORIGINAL"
                

                # Update the outer dictionary with the current batch's path-to-explanation mapping
                outer_extracted_dict.update(path_explanation_mapping)

                # Stop processing further batches if the current extracted dictionary contains exactly one key
                if len(extracted_dict) == 1:
                    print(f"Query {query_id}: Stopping, found exactly 1 key in batch {batch_index}.")
                    break  # Exit the loop when we find exactly one key
            else:
                print(f"Query {query_id}: Not enough images returned to create the grid in batch {batch_index}.")
                continue  # Proceed to the next batch if we don't have enough images

        # After processing, insert the accumulated results
        if outer_extracted_dict:
            # Prepare to insert the mappings (now using paths as keys)
            paths, explanations = zip(*outer_extracted_dict.items())

            print(f"Final Response for Query {query_id}: {response}")
            add_new_mappings_atomic(query_id, paths, explanations)
        else:
            print(f"Query {query_id}: No valid mappings to insert.")
    except Exception as e:
        print(f"An error occurred while processing Query {query_id}: {e}")


from concurrent.futures import ThreadPoolExecutor, as_completed

# Maximum number of worker threads
MAX_THREADS = 1

def main():
    threads = []
    executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

    for query_id_m_1, (_, query, original_image_filename) in enumerate(queries):
        query_id = query_id_m_1 + 1

        if has_entries(query_id):
            print(f"Query {query_id} already has entries, skipping...")
            continue

        # Retrieve paths (cannot be parallelized)
        retrieved_paths = get_retrieved_paths(query, original_image_filename)

        # Submit the processing to the thread pool
        future = executor.submit(process_query, query_id, query, original_image_filename, retrieved_paths)
        threads.append(future)

    # Wait for all threads to complete
    for future in as_completed(threads):
        try:
            future.result()
        except Exception as exc:
            print(f'Thread generated an exception: {exc}')

    executor.shutdown(wait=True)
    print("All queries have been processed.")

if __name__ == "__main__":
    main()
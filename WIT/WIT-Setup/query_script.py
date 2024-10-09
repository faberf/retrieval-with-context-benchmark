import os
import json
import requests

# Base Retriever class using abc
from abc import ABC, abstractmethod


class Retriever(ABC):
    def __init__(self, schema_name, host):
        self.schema_name = schema_name
        self.host = host

    @abstractmethod
    def make_payload(self, input_text):
        pass

    def query_files(self, input_text):
        url = f"{self.host}/api/{self.schema_name}/query"
        payload = self.make_payload(input_text)
        headers = {'Content-Type': 'application/json'}

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
                        filenames.append(file_name)
            return filenames  # Return all filenames to ensure ranks are available

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return []


# ClipRetriever uses baseline schema
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
                    "limit": "1000"
                },
                "local": {"filelookup": {"field": "file", "keys": "path"}}
            },
            "output": "filelookup"
        }


# CaptionDenseRetriever for no-metadata and with-metadata
class CaptionDenseRetriever(Retriever):
    def make_payload(self, input_text):
        return {
            "inputs": {
                "mytext1": {"type": "TEXT", "data": input_text}
            },
            "operations": {
                "captionDense1": {"type": "RETRIEVER", "field": "captionDense", "input": "mytext1"},
                "filelookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "captionDense1"}
            },
            "context": {
                "global": {
                    "limit": "1000"
                },
                "local": {"filelookup": {"field": "file", "keys": "path"}}
            },
            "output": "filelookup"
        }


# Load checkpoint file if it exists
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


# Save checkpoint progress
def save_checkpoint(processed_items, checkpoint_file):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_items), f, ensure_ascii=False, indent=4)


# Function to augment each item with retrieval results
def augment_item(item, clip_retriever, caption_retriever_no_metadata, caption_retriever_with_metadata):
    image_filename = os.path.splitext(item["image_file_name"])[0]  # Extract the base filename (without extension)

    # Perform the three retrievals, ensuring all results are available
    clip_top10 = clip_retriever.query_files(item["query"])[:10]  # Limit to top 10 for the final output
    no_metadata_top10 = caption_retriever_no_metadata.query_files(item["query"])[:10]
    metadata_top10 = caption_retriever_with_metadata.query_files(item["query"])[:10]

    # Get the rank of the original image file in each retrieval result
    clip_rank = clip_retriever.query_files(item["query"]).index(image_filename) + 1 if image_filename in clip_retriever.query_files(item["query"]) else None
    nometadata_rank = caption_retriever_no_metadata.query_files(item["query"]).index(image_filename) + 1 if image_filename in caption_retriever_no_metadata.query_files(item["query"]) else None
    metadata_rank = caption_retriever_with_metadata.query_files(item["query"]).index(image_filename) + 1 if image_filename in caption_retriever_with_metadata.query_files(item["query"]) else None

    # Augment the item with new fields
    item["clip_top10"] = clip_top10
    item["clip_rank"] = clip_rank
    item["nometadata_top10"] = no_metadata_top10
    item["nometadata_rank"] = nometadata_rank
    item["metadata_top10"] = metadata_top10
    item["metadata_rank"] = metadata_rank

    return item


# Function to process the file in batches and save results incrementally
def process_queries(input_file, output_file, checkpoint_file, batch_size=1):  # Batch size set to 1
    # Load input file
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Load checkpoint progress
    processed_queries = load_checkpoint(checkpoint_file)

    # Filter out already processed queries
    queries_to_process = [q for q in queries if q["image_id"] not in processed_queries]

    # Initialize retrievers
    clip_retriever = ClipRetriever(schema_name="baseline", host="http://localhost:7070")
    caption_retriever_no_metadata = CaptionDenseRetriever(schema_name="no-metadata", host="http://localhost:7070")
    caption_retriever_with_metadata = CaptionDenseRetriever(schema_name="with-metadata", host="http://localhost:7070")

    # Check if output file exists and handle its initial state
    if os.path.exists(output_file):
        if os.path.getsize(output_file) == 0:
            # If the file exists but is empty, initialize the JSON array
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
            needs_comma = False
        else:
            needs_comma = True  # If file already has content, append with commas
    else:
        # If file does not exist, create and start the JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
        needs_comma = False

    # Process the queries in batches of 1
    for i in range(0, len(queries_to_process), batch_size):
        batch = queries_to_process[i:i + batch_size]
        augmented_batch = []

        # Augment each query with the retrieval results
        for item in batch:
            augmented_item = augment_item(item, clip_retriever, caption_retriever_no_metadata, caption_retriever_with_metadata)
            augmented_batch.append(augmented_item)

        # Write augmented batch to output file
        with open(output_file, 'a', encoding='utf-8') as f:
            if needs_comma:
                f.write(',\n')  # Append comma if file already has content
            f.write(",\n".join(json.dumps(q, ensure_ascii=False, indent=4) for q in augmented_batch))
            needs_comma = True  # Subsequent entries should be comma-separated

        # Save progress to checkpoint
        processed_queries.update([q["image_id"] for q in batch])
        save_checkpoint(processed_queries, checkpoint_file)

    # Close the JSON array in the output file after all queries are processed
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')  # Close the JSON array

    print(f"Processing complete. Results saved to {output_file}")


# Example usage
input_file = "output_queries_improved.json"
output_file = "augmented_output.json"
checkpoint_file = "processing_checkpoint.json"

process_queries(input_file, output_file, checkpoint_file)

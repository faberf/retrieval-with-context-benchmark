import os
import json
import requests
import psycopg2
from abc import ABC, abstractmethod
import time
import math

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

conn.autocommit = True
cur = conn.cursor()

# Base Retriever class using abc
class Retriever(ABC):
    def __init__(self, schema_name, host, max_retries=1000, retry_delay=2):
        self.schema_name = schema_name
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
                            filenames.append(file_name)
                
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
                    "limit": "1000"
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

# CaptionDenseRetriever class
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
                    "limit": "1000"
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

# ClipDenseCaptionFusionRetriever class
class ClipDenseCaptionFusionRetriever(Retriever):
    def __init__(self, schema_name, host, clip_weight=0.5, caption_dense_weight=0.5, p=1.0, max_retries=1000, retry_delay=2):
        assert clip_weight + caption_dense_weight == 1.0, "Weights should sum to 1.0"
        self.clip_weight = clip_weight
        self.caption_dense_weight = caption_dense_weight
        self.p = p
        super().__init__(schema_name, host, max_retries, retry_delay)

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
                    "limit": "1000"
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

# Function to create tables if they don't exist for each retriever
def create_retriever_tables(retrievers):
    for retriever in retrievers:
        create_table_query = retriever.create_table_query()
        cur.execute(create_table_query)
        print(f"Table for {retriever.schema_name} ensured to exist.")

# Function to check which retrievers still need to process a query
def get_remaining_retrievers_for_query(query_id, retrievers):
    remaining_retrievers = []
    for retriever in retrievers:
        if not retriever.check_existing_result(query_id):
            remaining_retrievers.append(retriever)
    return remaining_retrievers

# Function to augment each item with retrieval results and store in the database
def augment_item_db(item, retrievers):
    image_filename = os.path.splitext(item["image_file_name"])[0]  # Extract the base filename (without extension)

    for retriever in retrievers:
        results = retriever.query_files(item["query"])
        ranking = results  # Store all results

        # Call retriever-specific method to add results to the database
        retriever.add_result_to_db(item["query_id"], ranking)

    return item

# Main processing function
def process_queries_db():
    # Initialize retrievers as a list
    host = "http://localhost:7070"
    retrievers = [
        ClipRetriever(schema_name="baseline", host=host),
        ClipRetriever(schema_name="clipvitl14", host=host),
        CaptionDenseRetriever(schema_name="no-metadata", host=host),
        CaptionDenseRetriever(schema_name="with-metadata", host=host),
        CaptionDenseRetriever(schema_name="two-categories", host=host),
        CaptionDenseRetriever(schema_name="full-metadata", host=host),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.9, caption_dense_weight=0.1),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.8, caption_dense_weight=0.2),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.7, caption_dense_weight=0.3),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.6, caption_dense_weight=0.4),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.5, caption_dense_weight=0.5),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.4, caption_dense_weight=0.6),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.3, caption_dense_weight=0.7),
        
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.9, caption_dense_weight=0.1),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.8, caption_dense_weight=0.2),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.7, caption_dense_weight=0.3),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.6, caption_dense_weight=0.4),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.5, caption_dense_weight=0.5),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.4, caption_dense_weight=0.6),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", host=host, clip_weight=0.3, caption_dense_weight=0.7),
        
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.9, caption_dense_weight=0.1),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.8, caption_dense_weight=0.2),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.7, caption_dense_weight=0.3),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.6, caption_dense_weight=0.4),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.5, caption_dense_weight=0.5),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.4, caption_dense_weight=0.6),
        ClipDenseCaptionFusionRetriever(schema_name="no-metadata", host=host, clip_weight=0.3, caption_dense_weight=0.7),
        
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.9, caption_dense_weight=0.1),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.8, caption_dense_weight=0.2),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.7, caption_dense_weight=0.3),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.6, caption_dense_weight=0.4),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.5, caption_dense_weight=0.5),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.4, caption_dense_weight=0.6),
        ClipDenseCaptionFusionRetriever(schema_name="two-categories", host=host, clip_weight=0.3, caption_dense_weight=0.7)
    ]

    # Ensure tables exist
    create_retriever_tables(retrievers)

    # Fetch all queries and associated image filenames
    cur.execute("""
        SELECT q.query_id, q.query_text, i.image_file_name
        FROM image_query_schema.query q
        JOIN image_query_schema.image i ON q.original_image_id = i.image_id;
    """)
    queries = cur.fetchall()

    # Iterate over the queries
    for query in queries:
        query_id, query_text, image_file_name = query

        # Get remaining retrievers for this query
        remaining_retrievers = get_remaining_retrievers_for_query(query_id, retrievers)
        
        if not remaining_retrievers:
            print(f"All retrievers have processed query {query_id}. Skipping.")
            continue

        # Process the query with the remaining retrievers
        augment_item_db({
            "query_id": query_id,
            "query": query_text,
            "image_file_name": image_file_name
        }, remaining_retrievers)

    print("Processing complete.")

# Example usage
process_queries_db()

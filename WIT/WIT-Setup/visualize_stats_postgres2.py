import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import psycopg2  # Import psycopg2 for PostgreSQL connection
import requests
import time

# LaTeX-like plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 12,
})

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

from abc import ABC, abstractmethod
import math

# Base Retriever class using abc
class Retriever(ABC):
    def __init__(self, schema_name, host, method_name, max_retries=1000, retry_delay=2, retrieval_limit=1000):
        self.schema_name = schema_name
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retrieval_limit = retrieval_limit
        self.method_name = method_name
    
    def get_method_name(self):
        return self.method_name

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
    
    def __init__(self, schema_name, host, max_retries=1000, retry_delay=2, retrieval_limit=1000, method_name=None):
        if method_name is not None:
            method_name = method_name
        else:
            method_name = f"CLIP ({schema_name})"
        super().__init__(schema_name, host, method_name, max_retries, retry_delay, retrieval_limit)
    
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
                rank = math.inf
            ranks_by_query[query_id].append(rank)

        # Collect and sort ranks per query
        return {query_id: sorted(ranks) for query_id, ranks in ranks_by_query.items()}
    
class CaptionDenseRetriever(Retriever):
    def __init__(self, schema_name, host, max_retries=1000, retry_delay=2, retrieval_limit=1000, method_name=None):
        if method_name is not None:
            method_name = method_name
        else:
            method_name = f"CaptionDense ({schema_name})"
        super().__init__(schema_name, host, method_name, max_retries, retry_delay, retrieval_limit)
    
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
                rank = math.inf
            ranks_by_query[query_id].append(rank)

        # Collect and sort ranks per query
        return {query_id: sorted(ranks) for query_id, ranks in ranks_by_query.items()}


# ClipDenseCaptionFusionRetriever class
class ClipDenseCaptionFusionRetriever(Retriever):
    def __init__(self, schema_name, host, clip_weight=0.5, caption_dense_weight=0.5, p=1.0, max_retries=1000, retry_delay=2, retrieval_limit=1000, method_name=None):
        assert clip_weight + caption_dense_weight == 1.0, "Weights should sum to 1.0"
        self.clip_weight = clip_weight
        self.caption_dense_weight = caption_dense_weight
        self.p = p
        if method_name is not None:
            method_name = method_name
        else:
            method_name = f"Fusion ({schema_name}, weights={clip_weight}/{caption_dense_weight}, p={p})"
        
        super().__init__(schema_name, host, method_name, max_retries, retry_delay, retrieval_limit)

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
                    rank = math.inf
                ranks_by_query[query_id].append(rank)

            # Collect and sort ranks per query
            return {query_id: sorted(ranks) for query_id, ranks in ranks_by_query.items()}

from joblib import Memory
memory = Memory("cachedir")

# Function to load queries from the PostgreSQL database
@memory.cache
def load_queries(retrievers):
    # Query to fetch data from the query and image tables, and all related image IDs
    sql = """
    SELECT q.query_id, q.query_text, q.historic_period, q.location, q.traditional_customs_practices,
        q.goes_beyond_metadata_using_image_contents, q.references_individual,
        q.query_language, q.query_language_equals_metadata_language,
        ARRAY_AGG(qi.image_id::text) AS image_ids,  -- Collect all associated image IDs as a list
        ARRAY_AGG(i.metadata_language) AS metadata_languages  -- Collect all associated metadata languages as a list
    FROM image_query_schema.query q
    JOIN image_query_schema.query_image qi ON q.query_id = qi.query_id
    JOIN image_query_schema.image i ON qi.image_id = i.image_id
    GROUP BY q.query_id
    """
    cur.execute(sql)
    rows = cur.fetchall()

    # Create a list of dictionaries to match the structure of the JSON data
    queries = []
    for row in rows:
        query = {
            'query_id': row[0],
            'query_text': row[1],
            'historic_period': row[2],
            'location': row[3],
            'traditional_customs_practices': row[4],
            'goes_beyond_metadata_using_image_contents': row[5],
            'references_individual': row[6],
            'query_language': row[7],
            'query_language_equals_metadata_language': row[8],
            'image_ids': row[9],  # List of UUIDs converted to strings
            'metadata_languages': row[10]
        }
        if len(row[9]) > 0:
            queries.append(query)

    # Build a mapping from query_id to query dictionary for easy lookup
    query_dict = {q['query_id']: q for q in queries}

    for retriever in retrievers:
        results = retriever.get_results(cur)
        rank_key = retriever.get_rank_key()
        for query_id, rank_list in results.items():
            # Get the corresponding query dictionary
            if query_id in query_dict:  
                query_dict[query_id][rank_key] = rank_list

    method_keys = [retriever.get_rank_key() for retriever in retrievers]
    
    return [
        query for query in query_dict.values()
        if all(query.get(rank_key) and len(query[rank_key]) == len(query.get(rank_key)) for rank_key in method_keys)
    ]



# Function to get image IDs from the image table in the database
def get_image_ids():
    sql = "SELECT image_id::text FROM image_query_schema.image"
    cur.execute(sql)
    rows = cur.fetchall()
    image_ids = set(row[0] for row in rows)
    return image_ids

# Function to collapse languages outside the top 20 into "Other"
def collapse_language(lang, top_languages):
    return lang if lang in top_languages else 'Other'

# # Function to calculate area under the curve on log-scaled x-axis
# def calculate_log_area(cumulative_df):
#     # Ensure the DataFrame is sorted by rank
#     cumulative_df = cumulative_df.sort_values('rank')
#     cumulative_df = cumulative_df.reset_index(drop=True)
    
#     # Extract rank and cumulative percentage values
#     x_values = cumulative_df['rank'].values
#     y_values = cumulative_df['cumulative_percent'].values
    
#     # Compute log10 of x values
#     log_x_values = np.log10(x_values)
    
#     # Initialize total area
#     total_area = 0.0
    
#     # Iterate over intervals
#     for i in range(len(x_values) - 1):
#         # Compute difference in log10(x)
#         delta_log_x = log_x_values[i+1] - log_x_values[i]
        
#         # Compute average of y-values
#         avg_y = (y_values[i] + y_values[i+1]) / 2.0
        
#         # Compute area of trapezoid
#         area = avg_y * delta_log_x
        
#         # Add to total area
#         total_area += area
    
#     return total_area

from collections import Counter
import numpy as np
import pandas as pd


def ndcg(ranked_list):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for a given ranked list,
    considering all ranks (k set to infinity).
    
    Parameters:
    ranked_list (list of int): A list of ranks (1-based index) where each rank has relevance of 1.
    
    Returns:
    float: The NDCG value.
    """
    # Number of relevant items
    num_relevant = len(ranked_list)
    
    if num_relevant == 0:
        return 0.0  # No relevant items, NDCG is zero
    
    # Compute DCG for the actual ranking
    dcg = 0.0
    for rank in ranked_list:
        dcg += 1 / np.log2(rank + 1)
    
    # Compute Ideal DCG (IDCG) where all relevant items are ranked at the top positions
    idcg = 0.0
    for i in range(1, num_relevant + 1):
        idcg += 1 / np.log2(i + 1)
    
    # Calculate NDCG
    ndcg_value = dcg / idcg
    
    return ndcg_value

# Function to compute stats and prepare data for plotting
def compute_stats(queries, image_ids, retrievers):
    stats = {}
    
    # Extract method keys and names from retrievers
    method_keys = [retriever.get_rank_key() for retriever in retrievers]
    method_names = [retriever.get_method_name() for retriever in retrievers]
    
    print(f"Total queries with ranks from all methods: {len(queries)}")
    
    # Queries per image
    image_query_count = Counter()
    for query in queries:
        for image_id in query['image_ids']:
            image_query_count[image_id] += 1
    
    # Add images with zero queries to the count
    for image_id in image_ids:
        if image_id not in image_query_count:
            image_query_count[image_id] = 0
            
    
    
    # Count metadata flags
    flag_counts = {
        'historic_period': 0,
        'location': 0,
        'traditional_customs_practices': 0,
        'query_language_equals_metadata_language': 0,
        'goes_beyond_metadata_using_image_contents': 0,
        'references_individual': 0
    }
    
    # Initialize the co-occurrence matrix
    flag_names = list(flag_counts.keys())
    co_occurrence_matrix = np.zeros((len(flag_names), len(flag_names)))
    
    # # Initialize language confusion matrix
    # language_confusion = pd.crosstab(
    #     [q['query_language'] for q in queries], 
    #     [q['metadata_languages'] for q in queries]
    # )
    
    # # Get top 20 languages
    # top_languages = language_confusion.sum().nlargest(20).index.tolist()

    # # Collapse non-top languages into "Other"
    # collapsed_query_languages = [collapse_language(q['query_language'], top_languages) for q in queries]
    # collapsed_metadata_languages = [collapse_language(q['metadata_language'], top_languages) for q in queries]

    # # Generate confusion matrix
    # language_confusion_collapsed = pd.crosstab(collapsed_query_languages, collapsed_metadata_languages)

    # # Ensure both axes have the same categories
    # language_confusion_collapsed = language_confusion_collapsed.reindex(index=top_languages + ['Other'], columns=top_languages + ['Other'], fill_value=0)
    # # Collect reciprocal ranks
    # # method_reciprocal_ranks = {method_name: [] for method_name in method_names}
    
    for query in queries:
        flags = [query.get(flag, False) for flag in flag_counts]
        
        # Update flag counts and co-occurrence matrix
        for i, flag in enumerate(flag_counts):
            if flags[i]:
                flag_counts[flag] += 1
        for i in range(len(flags)):
            for j in range(i, len(flags)):
                if flags[i] and flags[j]:
                    co_occurrence_matrix[i, j] += 1
                    if i != j:
                        co_occurrence_matrix[j, i] += 1  # Ensure symmetry
    
    # Store all stats
    stats['image_query_count'] = image_query_count
    stats['flag_counts'] = flag_counts
    stats['co_occurrence_matrix'] = co_occurrence_matrix
    stats['flag_names'] = flag_names
    # stats['language_confusion_collapsed'] = language_confusion_collapsed
    
    # Prepare DataFrame with ranks and flags
    data_list = []
    
    data_list_no_ranks = []
    
    # Map original flags to readable labels
    positive_flag_labels = {
        'historic_period': 'References a Historic Period',
        'location': 'References a Location',
        'traditional_customs_practices': 'References Customs or Practices',
        # 'query_language_equals_metadata_language': 'Unilingual',
        'goes_beyond_metadata_using_image_contents': 'Content Queries',
        'references_individual': 'References an Individual',
    }
    
    negative_flag_labels = {
        'historic_period': 'Does not Reference a Historic Period',
        'location': 'Does not Reference a Location',
        'traditional_customs_practices': 'Does not Reference Customs or Practices',
        # 'query_language_equals_metadata_language': 'Cross-Lingual',
        'goes_beyond_metadata_using_image_contents': 'Metadata Queries',
        'references_individual': 'Does not Reference an Individual',
    }
    
    query_language_counter = Counter([q['query_language'] for q in queries])
    top_5_query_languages = next(zip(*query_language_counter.most_common(5)))
    top_20_query_languages = next(zip(*query_language_counter.most_common(20)))
    
    for query in queries:
        # Ensure flags are booleans
        flags = {positive_flag_labels[flag]: bool(query.get(flag, False)) for flag in positive_flag_labels}
        flags.update({negative_flag_labels[flag]: not bool(query.get(flag, False)) for flag in negative_flag_labels})
        query_id = query['query_id']
        
        flags["query_language"] = query['query_language']
        flags["query_language_top5"] = query['query_language'] if query['query_language'] in top_5_query_languages else 'Other'
        flags["query_language_top20"] = query['query_language'] if query['query_language'] in top_20_query_languages else 'Other'
        
        metadata_language_counter = Counter(query['metadata_languages'])
        most_common_metadata_language, most_common_metadata_language_count = metadata_language_counter.most_common(1)[0]
        most_common_metadata_language_frequency = most_common_metadata_language_count / len(query['metadata_languages'])
        CUTOFF = 2/3.0
        if most_common_metadata_language_frequency >= CUTOFF:
            flags['most_common_metadata_language'] = most_common_metadata_language
            flags['most_common_metadata_language_top5'] = most_common_metadata_language if most_common_metadata_language in top_5_query_languages else 'Other'
            flags['most_common_metadata_language_top20'] = most_common_metadata_language if most_common_metadata_language in top_20_query_languages else 'Other'
        else:
            flags['most_common_metadata_language'] = 'Mixed'
            flags["most_common_metadata_language_top5"] = 'Mixed'
            flags["most_common_metadata_language_top20"] = 'Mixed'
        
        if flags['most_common_metadata_language'] == query['query_language']:
            flags["Unilingual"] = True
            flags["Cross-Lingual"] = False
        else:
            flags["Unilingual"] = False
            flags["Cross-Lingual"] = True
        
        data_list_no_ranks.append({"query_id": query_id, **flags})
        
    
        # For each method, collect rank and flags
        for rank_key, method_name in zip(method_keys, method_names):
            ranks = query.get(rank_key)
            if ranks and len(ranks) > 0:
                ndcg_value = ndcg(ranks)
                data = {
                    'method': method_name,
                    'worst_rank': ranks[-1],
                    "best_rank": ranks[0],
                    'ndcg': ndcg_value,
                    'query_id': query_id
                }
                data.update(flags)
                data_list.append(data)
                
    
    # Convert to DataFrame
    mrr_df = pd.DataFrame(data_list)
    
    base_df = pd.DataFrame(data_list_no_ranks)
    mrr_df['method'] = pd.Categorical(mrr_df['method'], categories=method_names, ordered=True)
    
    crosstab = pd.crosstab(base_df["query_language_top20"], base_df["most_common_metadata_language_top20"])
    
    #reindex using language counter
    crosstab = crosstab.reindex(index=top_20_query_languages + ('Other',), columns=top_20_query_languages + ('Other', "Mixed"), fill_value=0)
    
    
    # Store additional stats
    stats['mrr_df'] = mrr_df
    stats['positive_flag_labels'] = positive_flag_labels
    stats['negative_flag_labels'] = negative_flag_labels
    stats['top_5_query_languages'] = top_5_query_languages
    stats['top_20_query_languages'] = top_20_query_languages
    stats['language_confusion'] = crosstab
    stats['method_order'] = method_names
    
    return stats


def plot_overall_cumulative(stats):
    mrr_df = stats['mrr_df']
    total_queries = len(mrr_df['query_id'].unique())

    # Plot for worst_rank
    cumulative_data_worst = []
    for method_name, method_group in mrr_df.groupby("method"):
        sorted_ranks = np.sort(method_group["worst_rank"])
        cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
        cumulative_data_worst.extend({
            "method": method_name,
            "worst_rank": rank,
            "cumulative_count": cum_count
        } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

    cumulative_df_worst = pd.DataFrame(cumulative_data_worst)
    cumulative_df_worst = cumulative_df_worst.groupby(['method', 'worst_rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
    total_counts_worst = cumulative_df_worst.groupby('method')['cumulative_count'].transform('max')
    cumulative_df_worst['cumulative_percent'] = (cumulative_df_worst['cumulative_count'] / total_counts_worst) * 100
    
    cumulative_df_worst["method"] = pd.Categorical(cumulative_df_worst["method"], categories=stats['method_order'], ordered=True)

    # Plotting for worst_rank
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cumulative_df_worst, x="worst_rank", y="cumulative_percent", hue="method", alpha=0.7, markers=True)
    plt.xscale("log")
    plt.xlabel("Worst Rank (log scale)")
    plt.ylabel("Cumulative Percentage (\%)")
    plt.title(f"Cumulative Frequencies of Worst Rank by Method ({total_queries} Queries Total)")
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

    # Plot for best_rank
    cumulative_data_best = []
    for method_name, method_group in mrr_df.groupby("method"):
        sorted_ranks = np.sort(method_group["best_rank"])
        cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
        cumulative_data_best.extend({
            "method": method_name,
            "best_rank": rank,
            "cumulative_count": cum_count
        } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

    cumulative_df_best = pd.DataFrame(cumulative_data_best)
    cumulative_df_best = cumulative_df_best.groupby(['method', 'best_rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
    total_counts_best = cumulative_df_best.groupby('method')['cumulative_count'].transform('max')
    cumulative_df_best['cumulative_percent'] = (cumulative_df_best['cumulative_count'] / total_counts_best) * 100
    
    cumulative_df_best["method"] = pd.Categorical(cumulative_df_best["method"], categories=stats['method_order'], ordered=True)

    # Plotting for best_rank
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cumulative_df_best, x="best_rank", y="cumulative_percent", hue="method", alpha=0.7, markers=True)
    plt.xscale("log")
    plt.xlabel("Best Rank (log scale)")
    plt.ylabel("Cumulative Percentage (\%)")
    plt.title(f"Cumulative Frequencies of Best Rank by Method ({total_queries} Queries Total)")
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

    # Plot for ndcg
    cumulative_data_ndcg = []
    for method_name, method_group in mrr_df.groupby("method"):
        # Sort ndcg scores in descending order
        sorted_ndcg = np.sort(method_group["ndcg"])[::-1]  # Reverse the sorted array
        cumulative_counts = np.arange(1, len(sorted_ndcg) + 1)
        cumulative_percentages = (cumulative_counts / len(sorted_ndcg)) * 100
        cumulative_data_ndcg.extend({
            "method": method_name,
            "ndcg": ndcg_score,
            "cumulative_percent": cum_percent
        } for ndcg_score, cum_percent in zip(sorted_ndcg, cumulative_percentages))

    cumulative_df_ndcg = pd.DataFrame(cumulative_data_ndcg)
    
    cumulative_df_ndcg["method"] = pd.Categorical(cumulative_df_ndcg["method"], categories=stats['method_order'], ordered=True)

    # Plotting for ndcg
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cumulative_df_ndcg, x="ndcg", y="cumulative_percent", hue="method", alpha=0.7, markers=True)
    plt.gca().invert_xaxis()  # Reverse the x-axis to go from 1 to 0
    plt.xlabel("nDCG")
    plt.ylabel("Cumulative Percentage (\%)")
    plt.title(f"Cumulative Frequencies of nDCG by Method ({total_queries} Queries Total)")
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=mrr_df, x='method', y='ndcg', ci='sd', capsize=0.2)
    plt.xlabel("Method")
    plt.ylabel("Average nDCG")
    plt.title("Average nDCG by Method")
    plt.tight_layout()
    plt.show()
    
    method_order = stats['method_order']

    # Compute mean and standard deviation per method
    stats_ndcg = mrr_df.groupby('method', observed=True)['ndcg'].agg(['mean', 'std']).reset_index()

    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Shifted x-values
    x_shift = 0.25
    x_positions = [i + x_shift for i in range(len(method_order))]

    # Plot the violin plot
    sns.violinplot(
        data=mrr_df,
        x='method',
        y='ndcg',
        order=method_order,
        cut=0,
        inner=None,
        scale='width',
        linewidth=1,
        ax=ax
    )

    # Modify the violins to display only the right half and apply the x_shift
    for i, artist in enumerate(ax.collections):
        paths = artist.get_paths()
        for path in paths:
            vertices = path.vertices
            x_center = np.mean(vertices[:, 0])
            vertices[:, 0] = np.maximum(vertices[:, 0], x_center)  # Keep only the right half
            vertices[:, 0] += x_shift  # Apply the x_shift to each violin

    # Add reduced vertical jitter to y-values to prevent overplotting
    jittered_ndcg = mrr_df['ndcg'] + np.random.uniform(-0.005, 0.005, size=len(mrr_df))

    # Plot the strip plot (raincloud) on the left side with increased horizontal jitter (0.2)
    strip = sns.stripplot(
        x=mrr_df['method'],
        y=jittered_ndcg,
        order=method_order,
        color='k',
        size=3,
        jitter=0.2,  # Increased jitter to 0.2
        alpha=0.6,
        ax=ax
    )

    # Shift the strip plot to the left side and apply the x_shift
    for collection in strip.collections:
        offsets = collection.get_offsets()
        offsets[:, 0] = offsets[:, 0] - 0.22 + x_shift  # Shift x-values by 0.25
        collection.set_offsets(offsets)

    # Overlay the mean points with larger error bars, applying the x_shift
    ax.errorbar(
        x=[i + x_shift for i in range(len(stats_ndcg))],  # Shift x-values by 0.25
        y=stats_ndcg['mean'],
        yerr=stats_ndcg['std'],
        fmt='o',
        color='red',
        capsize=12,      # Increased capsize
        capthick=2,
        elinewidth=0,    # Set to zero for cap-only error bars
        markersize=4,    # Adjusted marker size
        zorder=10
    )

    # Set y-axis limits with padding
    ax.set_ylim(-0.05, 1.05)

    # Shift the x-axis labels (tick labels) by 0.25
    ax.set_xticks([i + x_shift for i in range(len(method_order))])
    ax.set_xticklabels(method_order)

    # Labels and title
    plt.xlabel("Method")
    plt.ylabel("nDCG")
    plt.title(f"nDCG Distribution by Method ({total_queries} Queries Total)")

    # Calculate the specific nDCG values
    ndcg_rank1 = 1.0
    ndcg_rank2 = 1.0 / np.log2(3)         # Approximately 0.63093
    ndcg_rank3 = 1.0 / np.log2(4)         # 0.5
    ndcg_rank10 = 1.0 / np.log2(11)       # Approximately 0.28854
    ndcg_rank100 = 1.0 / np.log2(101)     # Approximately 0.15056

    # Define the x-range for the horizontal lines
    x_start = -0.5    # Start at the left edge of the x-axis
    x_end = -0.3       # Shortened x-range for the horizontal lines

    # Add horizontal lines at the specified nDCG values with limited x-range
    ax.hlines(y=ndcg_rank1, xmin=x_start, xmax=x_end, color='blue', linestyle='--', linewidth=1)
    ax.hlines(y=ndcg_rank2, xmin=x_start, xmax=x_end, color='green', linestyle='--', linewidth=1)
    ax.hlines(y=ndcg_rank3, xmin=x_start, xmax=x_end, color='purple', linestyle='--', linewidth=1)
    ax.hlines(y=ndcg_rank10, xmin=x_start, xmax=x_end, color='orange', linestyle='--', linewidth=1)
    ax.hlines(y=ndcg_rank100, xmin=x_start, xmax=x_end, color='brown', linestyle='--', linewidth=1)

    # Adjust x-axis limits to prevent labels from being cut off
    ax.set_xlim(-0.5, len(method_order) - 0.2 + x_shift)

    # Define label positions and colors
    label_positions = {
        ndcg_rank1: {'text': 'Rank 1', 'color': 'blue', 'y_offset': -0.02},
        ndcg_rank2: {'text': 'Rank 2', 'color': 'green', 'y_offset': 0.01},
        ndcg_rank3: {'text': 'Rank 3', 'color': 'purple', 'y_offset': 0.01},
        ndcg_rank10: {'text': 'Rank 10', 'color': 'orange', 'y_offset': 0.01},
        ndcg_rank100: {'text': 'Rank 100', 'color': 'brown', 'y_offset': 0.01},
    }

    # Add labels next to the lines, close to the left border
    for ndcg_value, info in label_positions.items():
        ax.text(
            x=x_start + 0.02,  # Slightly to the right of x_start
            y=ndcg_value + info['y_offset'],
            s=info['text'],
            color=info['color'],
            fontsize=9,
            ha='left',
            va='bottom' if info['y_offset'] >= 0 else 'top'
        )

    # Show the plot
    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_per_flag_cumulative(stats, flags_group):
    mrr_df = stats['mrr_df']

    # Plot for best_rank
    fig_best, axes_best = plt.subplots(len(flags_group)//2, 2, figsize=(15, 5 * len(flags_group)//2))
    axes_best = axes_best.ravel()

    # Plot for worst_rank
    fig_worst, axes_worst = plt.subplots(len(flags_group)//2, 2, figsize=(15, 5 * len(flags_group)//2))
    axes_worst = axes_worst.ravel()

    # Plot for ndcg
    fig_ndcg, axes_ndcg = plt.subplots(len(flags_group)//2, 2, figsize=(15, 5 * len(flags_group)//2))
    axes_ndcg = axes_ndcg.ravel()

    for i, flag_row in enumerate(zip(flags_group[::2], flags_group[1::2])):
        for j, flag in enumerate(flag_row):
            df_filtered = mrr_df[mrr_df[flag] == True]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                idx = i*2 + j
                # For best_rank
                ax = axes_best[idx]
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                ax.set_title(flag)
                ax.set_axis_off()

                # For worst_rank
                ax = axes_worst[idx]
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                ax.set_title(flag)
                ax.set_axis_off()

                # For ndcg
                ax = axes_ndcg[idx]
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                ax.set_title(flag)
                ax.set_axis_off()
                continue

            # -----------------------
            # Plot for best_rank
            # -----------------------
            cumulative_data_best = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["best_rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_data_best.extend({
                    "method": method_name,
                    "best_rank": rank,
                    "cumulative_count": cum_count
                } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

            cumulative_df_best = pd.DataFrame(cumulative_data_best)
            cumulative_df_best = cumulative_df_best.groupby(['method', 'best_rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
            total_counts_best = cumulative_df_best.groupby('method')['cumulative_count'].transform('max')
            cumulative_df_best['cumulative_percent'] = (cumulative_df_best['cumulative_count'] / total_counts_best) * 100

            ax_best = axes_best[i*2 + j]
            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df_best[cumulative_df_best['method'] == method_name]
                ax_best.plot(method_data['best_rank'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            ax_best.set_title(f'{flag} ({total_queries} Queries Total)')
            ax_best.set_xscale('log')
            ax_best.set_xlabel('Best Rank (log scale)')
            ax_best.set_ylabel('Cumulative Percentage (\%)')
            ax_best.legend()

            # -----------------------
            # Plot for worst_rank
            # -----------------------
            cumulative_data_worst = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["worst_rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_data_worst.extend({
                    "method": method_name,
                    "worst_rank": rank,
                    "cumulative_count": cum_count
                } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

            cumulative_df_worst = pd.DataFrame(cumulative_data_worst)
            cumulative_df_worst = cumulative_df_worst.groupby(['method', 'worst_rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
            total_counts_worst = cumulative_df_worst.groupby('method')['cumulative_count'].transform('max')
            cumulative_df_worst['cumulative_percent'] = (cumulative_df_worst['cumulative_count'] / total_counts_worst) * 100

            ax_worst = axes_worst[i*2 + j]
            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df_worst[cumulative_df_worst['method'] == method_name]
                ax_worst.plot(method_data['worst_rank'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            ax_worst.set_title(f'{flag} ({total_queries} Queries Total)')
            ax_worst.set_xscale('log')
            ax_worst.set_xlabel('Worst Rank (log scale)')
            ax_worst.set_ylabel('Cumulative Percentage (\%)')
            ax_worst.legend()

            # -----------------------
            # Plot for ndcg
            # -----------------------
            cumulative_data_ndcg = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ndcg = np.sort(method_group["ndcg"])[::-1]  # Sort ndcg in descending order
                cumulative_counts = np.arange(1, len(sorted_ndcg) + 1)
                cumulative_percentages = (cumulative_counts / len(sorted_ndcg)) * 100
                cumulative_data_ndcg.extend({
                    "method": method_name,
                    "ndcg": ndcg_score,
                    "cumulative_percent": cum_percent
                } for ndcg_score, cum_percent in zip(sorted_ndcg, cumulative_percentages))

            cumulative_df_ndcg = pd.DataFrame(cumulative_data_ndcg)

            ax_ndcg = axes_ndcg[i*2 + j]
            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df_ndcg[cumulative_df_ndcg['method'] == method_name]
                ax_ndcg.plot(method_data['ndcg'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            ax_ndcg.set_title(f'{flag} ({total_queries} Queries Total)')
            ax_ndcg.set_xlabel('nDCG')
            ax_ndcg.set_ylabel('Cumulative Percentage (\%)')
            ax_ndcg.legend()
            ax_ndcg.invert_xaxis()  # Reverse the x-axis to go from 1 to 0

    fig_best.suptitle("Cumulative Frequencies of Best Rank by Method and Query Property")
    fig_best.tight_layout(pad=2, rect=[0, 0.03, 1, 0.95])
    

    fig_worst.suptitle(f"Cumulative Frequencies of Worst Rank by Method and Query Property")
    fig_worst.tight_layout(pad=2, rect=[0, 0.03, 1, 0.95])

    fig_ndcg.suptitle(f"Cumulative Frequencies of nDCG by Method and Query Property")
    fig_ndcg.tight_layout(pad=2, rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_language_cumulative(stats, language_dict):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    mrr_df = stats['mrr_df']

    # The languages to be plotted come from the keys of the dictionary
    # and their corresponding display names come from the values
    language_groups = list(language_dict.keys())

    # List of metrics to plot
    metrics = ['best_rank', 'worst_rank', 'ndcg']

    for metric in metrics:
        if metric == 'best_rank':
            metric_title = 'Best Rank'
        elif metric == 'worst_rank':
            metric_title = 'Worst Rank'
        elif metric == 'ndcg':
            metric_title = 'nDCG'
        num_languages = len(language_groups)
        # Create subplots: rows = number of languages, columns = 2 (Unilingual and Crosslingual)
        fig, axes = plt.subplots(num_languages, 2, figsize=(15, 5 * num_languages))
        axes = axes.ravel()

        for i, lang_key in enumerate(language_groups):
            display_name = language_dict[lang_key]  # Get the display name from the dictionary
            for j, linguicity in enumerate(["Unilingual", "Cross-Lingual"]):  # True for Unilingual, False for Crosslingual
                idx = i * 2 + j
                ax = axes[idx]

                # Regular filtering for the specified language
                df_filtered = mrr_df[
                    (mrr_df['query_language_top5'] == lang_key) &
                    (mrr_df[linguicity])
                ]

                total_queries = len(df_filtered['query_id'].unique())

                if df_filtered.empty:
                    ax.text(0.5, 0.5, 'No data available',
                            horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f'{linguicity} {display_name} ({total_queries} Queries Total)')
                    ax.set_axis_off()
                    continue

                cumulative_data = []
                for method_name, method_group in df_filtered.groupby("method"):
                    if metric in ['best_rank', 'worst_rank']:
                        sorted_values = np.sort(method_group[metric])
                        cumulative_counts = np.arange(1, len(sorted_values) + 1)
                        cumulative_data.extend({
                            "method": method_name,
                            metric: value,
                            "cumulative_count": cum_count
                        } for value, cum_count in zip(sorted_values, cumulative_counts))
                    elif metric == 'ndcg':
                        sorted_values = np.sort(method_group[metric])[::-1]  # Descending order
                        cumulative_counts = np.arange(1, len(sorted_values) + 1)
                        cumulative_percentages = (cumulative_counts / len(sorted_values)) * 100
                        cumulative_data.extend({
                            "method": method_name,
                            metric: value,
                            "cumulative_percent": cum_percent
                        } for value, cum_percent in zip(sorted_values, cumulative_percentages))

                cumulative_df = pd.DataFrame(cumulative_data)
                if metric in ['best_rank', 'worst_rank']:
                    cumulative_df = cumulative_df.groupby(['method', metric]).size() \
                        .groupby(level=0).cumsum().reset_index(name='cumulative_count')
                    total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
                    cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

                for method_name in df_filtered['method'].unique():
                    method_data = cumulative_df[cumulative_df['method'] == method_name]
                    ax.plot(method_data[metric], method_data['cumulative_percent'],
                            label=method_name, alpha=0.7)

                ax.set_title(f'{linguicity} {display_name} ({total_queries} Queries Total)')

                if metric in ['best_rank', 'worst_rank']:
                    ax.set_xscale('log')
                elif metric == 'ndcg':
                    ax.invert_xaxis()  # Reverse x-axis for nDCG
                ax.set_xlabel(metric_title)

                ax.set_ylabel('Cumulative Percentage (\%)')
                ax.legend()

        figure_title = "Cumulative Frequencies of " + metric_title + " by Language"
        fig.suptitle(f'{figure_title}')
        plt.tight_layout(pad=2, rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_per_flag_bar(stats, flags_group, figure_title):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    mrr_df = stats['mrr_df']
    method_names = stats['method_order']

    data = []

    for flag in flags_group:
            
            df_filtered = mrr_df[mrr_df[flag] == True]
            if not df_filtered.empty:
                for method in method_names:
                    df_method = df_filtered[df_filtered['method'] == method]
                    if not df_method.empty:
                        # Collect individual nDCG values
                        for ndcg_value in df_method['ndcg']:
                            data.append({
                                'method': method,
                                'flag_label': flag,
                                'ndcg': ndcg_value
                            })

    df_data = pd.DataFrame(data)

    # Function to split long labels
    def split_label(label, max_length=20):
        if len(label) <= max_length:
            return label
        else:
            mid = len(label) // 2
            # Find nearest space to mid
            split_pos = label.rfind(' ', 0, mid)
            if split_pos == -1:
                split_pos = label.find(' ', mid)
                if split_pos == -1:
                    split_pos = mid
            part1 = label[:split_pos]
            part2 = label[split_pos:].strip()
            return part1 + '\n' + part2

    df_data['flag_label'] = df_data['flag_label'].apply(split_label)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_data,
        x='flag_label',
        y='ndcg',
        hue='method',
        estimator=np.mean,
        ci='sd',
        capsize=0.2
    )
    plt.xlabel('Query Property')
    plt.ylabel('Average nDCG')
    plt.title(figure_title)
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_language_bar(stats, language_dict):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    mrr_df = stats['mrr_df']
    method_names = stats['method_order']

    data = []

    # The languages to be plotted come from the keys of the dictionary
    # and their corresponding display names come from the values
    language_groups = list(language_dict.keys())

    for i, lang_key in enumerate(language_groups):
        display_name = language_dict[lang_key]  # Get the display name from the dictionary
        for j, linguicity in enumerate(["Unilingual", "Cross-Lingual"]):  # True for Unilingual, False for Crosslingual

            # Regular filtering for the specified language
            df_filtered = mrr_df[
                (mrr_df['query_language_top5'] == lang_key) &
                (mrr_df[linguicity])
            ]
            lang_label = f'{display_name} ({linguicity})'

            if not df_filtered.empty:
                for method in method_names:
                    df_method = df_filtered[df_filtered['method'] == method]
                    if not df_method.empty:
                        # Collect individual nDCG values
                        for ndcg_value in df_method['ndcg']:
                            data.append({
                                'method': method,
                                'language': lang_label,
                                'ndcg': ndcg_value
                            })

    df_data = pd.DataFrame(data)

    # Function to split long labels
    def split_label(label, max_length=20):
        if len(label) <= max_length:
            return label
        else:
            mid = len(label) // 2
            # Find nearest space to mid
            split_pos = label.rfind(' ', 0, mid)
            if split_pos == -1:
                split_pos = label.find(' ', mid)
                if split_pos == -1:
                    split_pos = mid
            part1 = label[:split_pos]
            part2 = label[split_pos:].strip()
            return part1 + '\n' + part2

    df_data['language'] = df_data['language'].apply(split_label)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_data,
        x='language',
        y='ndcg',
        hue='method',
        estimator=np.mean,
        ci='sd',
        capsize=0.2
    )
    plt.xlabel('Language')
    plt.ylabel('Average nDCG')
    plt.title('Average nDCG by Method and Language')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




def plot_language_confusion_matrix(stats):
    # Get the language confusion matrix
    language_confusion = stats['language_confusion']

    # # Sum over query languages to get frequencies of metadata languages
    # metadata_lang_totals = language_confusion.sum(axis=0)
    
    # # Sort the metadata languages by frequency in descending order
    # sorted_metadata_langs = metadata_lang_totals.sort_values(ascending=False).index.tolist()

    # # Check if "Other" is in the sorted list, and if so, move it to the end
    # if "Other" in sorted_metadata_langs:
    #     sorted_metadata_langs.remove("Other")
    #     sorted_metadata_langs.append("Other")

    # Reorder both rows and columns of the confusion matrix based on sorted metadata languages
    # language_confusion_sorted = language_confusion.reindex(index=sorted_metadata_langs, columns=sorted_metadata_langs)

    plt.figure(figsize=(12, 10))
    sns.heatmap(language_confusion, annot=True, cmap='coolwarm', fmt='g')
    plt.title('Language Confusion Matrix (Top 20 Languages + Other)')
    plt.xlabel('Metadata Language')
    plt.ylabel('Query Language')
    plt.tight_layout()
    plt.show()


def plot_queries_per_image(stats):
    image_query_counts = list(stats['image_query_count'].values())
    total_images = len(stats['image_query_count'])

    plt.figure(figsize=(8, 6))
    plt.hist(image_query_counts, bins=20, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of Queries per Image (Total Images = {total_images})')
    plt.xlabel('Number of Queries')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.show()

def plot_flag_distribution(stats):
    flag_labels = stats['positive_flag_labels']
    # Map flag counts to readable labels
    flag_counts = {flag_labels[k]: v for k, v in stats['flag_counts'].items()}

    plt.figure(figsize=(8, 6))
    plt.bar(flag_counts.keys(), flag_counts.values(), color='green')
    plt.title('Counts of Metadata Flags in Queries')
    plt.xlabel('Metadata Flag')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_top_referenced_images(stats):
    top_images = stats['image_query_count'].most_common(10)
    image_ids, query_counts = zip(*top_images)

    plt.figure(figsize=(10, 6))
    plt.barh(image_ids, query_counts, color='orange')
    plt.title('Top 10 Referenced Images')
    plt.xlabel('Number of Queries')
    plt.ylabel('Image ID')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_cooccurrence_matrix(stats):
    flag_labels = stats['positive_flag_labels']
    # Map flag names to readable labels
    flag_names = [flag_labels[fn] for fn in stats['flag_names']]

    plt.figure(figsize=(8, 6))
    sns.heatmap(stats['co_occurrence_matrix'], annot=True, cmap='Blues', xticklabels=flag_names, yticklabels=flag_names)
    plt.title('Co-occurrence Matrix of Metadata Flags')
    plt.xlabel('Metadata Flags')
    plt.ylabel('Metadata Flags')
    plt.tight_layout()
    plt.show()

# Main function to execute the process
def main():
    # Initialize retrievers dynamically
    host = "http://localhost:7070"
    # retrievers = [
    #     ClipRetriever(schema_name="baseline", host=host, method_name="CLIP"),
    #     # # CaptionDenseRetriever(schema_name="no-metadata", host=host),
    #     CaptionDenseRetriever(schema_name="with-metadata", method_name="Ours", host=host),
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="Fusion", host=host, clip_weight=0.6, caption_dense_weight=0.4)
    # ]
    
    # retrievers = [
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="90\% Clip, 10\% Ours", host=host, clip_weight=0.9, caption_dense_weight=0.1),
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="70\% Clip, 30\% Ours", host=host, clip_weight=0.7, caption_dense_weight=0.3),
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="60\% Clip, 40\% Ours", host=host, clip_weight=0.6, caption_dense_weight=0.4), # good
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="50\% Clip, 50\% Ours", host=host, clip_weight=0.5, caption_dense_weight=0.5),
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="30\% Clip, 70\% Ours", host=host, clip_weight=0.3, caption_dense_weight=0.7),
    # ]
    
    # retrievers = [
    #     ClipRetriever(schema_name="baseline", host=host, method_name="Open Clip"),
    #     ClipRetriever(schema_name="clipvitl14", host=host, method_name="OpenAI Clip"),
    #     CaptionDenseRetriever(schema_name="with-metadata", method_name="Ours", host=host),
    #     ClipDenseCaptionFusionRetriever(schema_name="with-metadata", method_name="Ours + Open Clip", host=host, clip_weight=0.6, caption_dense_weight=0.4)
    # ]
    
    # retrievers = [
    #     ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="70\% Clip, 30\% Ours (All Metadata)", host=host, clip_weight=0.7, caption_dense_weight=0.3),
    #     ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="60\% Clip, 40\% Ours (All Metadata)", host=host, clip_weight=0.6, caption_dense_weight=0.4),
    #     ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="50\% Clip, 50\% Ours (All Metadata)", host=host, clip_weight=0.5, caption_dense_weight=0.5),
    #     ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="40\% Clip, 60\% Ours (All Metadata)", host=host, clip_weight=0.4, caption_dense_weight=0.6), # good
    #     ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="30\% Clip, 70\% Ours (All Metadata)", host=host, clip_weight=0.3, caption_dense_weight=0.7),
    # ]
    
    # retrievers = [
    #     ClipDenseCaptionFusionRetriever(schema_name="two-categories", method_name="70\% Clip, 30\% Ours (Two Categories)", host=host, clip_weight=0.7, caption_dense_weight=0.3),
    #     ClipDenseCaptionFusionRetriever(schema_name="two-categories", method_name="60\% Clip, 40\% Ours (Two Categories)", host=host, clip_weight=0.6, caption_dense_weight=0.4),
    #     ClipDenseCaptionFusionRetriever(schema_name="two-categories", method_name="50\% Clip, 50\% Ours (Two Categories)", host=host, clip_weight=0.5, caption_dense_weight=0.5), # good
    #     ClipDenseCaptionFusionRetriever(schema_name="two-categories", method_name="40\% Clip, 60\% Ours (Two Categories)", host=host, clip_weight=0.4, caption_dense_weight=0.6),
    #     ClipDenseCaptionFusionRetriever(schema_name="two-categories", method_name="30\% Clip, 70\% Ours (Two Categories)", host=host, clip_weight=0.3, caption_dense_weight=0.7),
    # ]
    
    retrievers = list(reversed([
        CaptionDenseRetriever(schema_name="no-metadata", method_name="Ours (No Metadata)" ,host=host),
        ClipRetriever(schema_name="baseline", host=host, method_name="CLIP"),
        CaptionDenseRetriever(schema_name="with-metadata", method_name="Ours (One Category)", host=host),
        CaptionDenseRetriever(schema_name="two-categories", method_name="Ours (Two Categories)", host=host),
        CaptionDenseRetriever(schema_name="full-metadata", method_name="Ours (All Metadata)", host=host),
        ClipDenseCaptionFusionRetriever(schema_name="full-metadata", method_name="Best Fusion (All Metadata + CLIP)", host=host, clip_weight=0.4, caption_dense_weight=0.6)
    ]))
    



    # Load the queries from the database
    queries = load_queries(retrievers)

    print(len(queries), "queries loaded.")

    if not queries:
        print("No queries loaded.")
        return

    # Get image IDs from the database
    image_ids = get_image_ids()

    # Compute stats
    stats = compute_stats(queries, image_ids, retrievers)

    # Now, split the flags into two groups of three
    flags_group1 = [
        'References a Historic Period',
        'Does not Reference a Historic Period',
        'References a Location',
        'Does not Reference a Location',
    ]
    flags_group2 = [
        'References Customs or Practices',
        'Does not Reference Customs or Practices',
        'Unilingual',
        'Cross-Lingual'
    ]
    flags_group3 = [
        'Metadata Queries',
        'Content Queries',
        'References an Individual',
        'Does not Reference an Individual'
    ]



    # Visualizations
    plot_overall_cumulative(stats)
    # plot_per_flag_cumulative(stats, flags_group1)
    # plot_per_flag_cumulative(stats, flags_group2)
    # plot_per_flag_cumulative(stats, flags_group3)

    # # Plot cumulative subplots for top 5 languages and 'Other'
    # plot_language_cumulative(stats, {"en":"English", "de":"German"})
    # plot_language_cumulative(stats, {"es":"Spanish", "it":"Italian"})
    # plot_language_cumulative(stats, {"ru":"Russian", "Other":"Other"})
    
    # plot_per_flag_bar(stats, flags_group1 + flags_group2 + flags_group3, 'Average nDCG by Query Property and Method')
    # plot_language_bar(stats, {"en":"English", "de": "German", "es":"Spanish", "it": "Italian", "ru":"Russian", "Other":"Other"})

    # # Other plotting functions
    plot_language_confusion_matrix(stats)
    # plot_queries_per_image(stats, queries)
    # plot_flag_distribution(stats, queries)
    # plot_top_referenced_images(stats, queries)
    # plot_cooccurrence_matrix(stats, queries)

    # # Close the database connection
    # cur.close()
    # conn.close()


# Run the main function
if __name__ == '__main__':
    main()

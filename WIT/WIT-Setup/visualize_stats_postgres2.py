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


# Function to load queries from the PostgreSQL database
def load_queries(retrievers):
    # Query to fetch data from the query and image tables, and all related image IDs
    sql = """
    SELECT q.query_id, q.query_text, q.historic_period, q.location, q.traditional_customs_practices,
        q.goes_beyond_metadata_using_image_contents, q.references_individual,
        q.query_language, q.query_language_equals_metadata_language,
        ARRAY_AGG(qi.image_id::text) AS image_ids,  -- Collect all associated image IDs as a list
        i.metadata_language
    FROM image_query_schema.query q
    JOIN image_query_schema.query_image qi ON q.query_id = qi.query_id
    JOIN image_query_schema.image i ON qi.image_id = i.image_id
    GROUP BY q.query_id, i.metadata_language
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
            'metadata_language': row[10]
        }
        queries.append(query)

    # Build a mapping from query_id to query dictionary for easy lookup
    query_dict = {q['query_id']: q for q in queries}

    for retriever in retrievers:
        results = retriever.get_results(cur)
        rank_key = retriever.get_rank_key()
        for query_id, rank_list in enumerate(results):
            # Get the corresponding query dictionary
            if query_id + 1 in query_dict:  # query_id is now 1-based in this iteration
                query = query_dict[query_id + 1]
                query[rank_key] = rank_list  # Directly assign the rank list

    return list(query_dict.values())



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

# Function to compute stats and prepare data for plotting
def compute_stats(queries, image_ids, retrievers):
    stats = {}
    
    # Extract method keys and names from retrievers
    method_keys = [retriever.get_rank_key() for retriever in retrievers]
    method_names = [retriever.get_method_name() for retriever in retrievers]
    
    # Filter queries to include those with ranks from all methods
    filtered_queries = [
        query for query in queries
        if all(query.get(rank_key) and len(query[rank_key]) > 0 for rank_key in method_keys)
    ]
    
    print(f"Total queries with ranks from all methods: {len(filtered_queries)}")
    
    # Use filtered_queries for stats computation
    queries = filtered_queries
    
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
    
    # Initialize language confusion matrix
    language_confusion = pd.crosstab(
        [q['query_language'] for q in queries], 
        [q['metadata_language'] for q in queries]
    )
    
    # Get top 20 languages
    top_languages = language_confusion.sum().nlargest(20).index.tolist()
    
    # Collapse non-top languages into "Other"
    collapsed_query_languages = [collapse_language(q['query_language'], top_languages) for q in queries]
    collapsed_metadata_languages = [collapse_language(q['metadata_language'], top_languages) for q in queries]
    language_confusion_collapsed = pd.crosstab(collapsed_query_languages, collapsed_metadata_languages)
    
    # Collect reciprocal ranks
    method_reciprocal_ranks = {method_name: [] for method_name in method_names}
    
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
    
        # Calculate reciprocal ranks and collect them
        for rank_key, method_name in zip(method_keys, method_names):
            ranks = query.get(rank_key)
            if ranks and len(ranks) > 0:
                worst_rank = ranks[-1]  # Worst case rank (highest number)
                method_reciprocal_ranks[method_name].append(1.0 / worst_rank)
            else:
                method_reciprocal_ranks[method_name].append(0.0)
    
    # Store all stats
    stats['image_query_count'] = image_query_count
    stats['flag_counts'] = flag_counts
    stats['co_occurrence_matrix'] = co_occurrence_matrix
    stats['flag_names'] = flag_names
    stats['language_confusion_collapsed'] = language_confusion_collapsed
    stats['method_reciprocal_ranks'] = method_reciprocal_ranks
    
    # Prepare DataFrame with ranks and flags
    data_list = []
    flags_list = list(flag_counts.keys())
    
    # Map original flags to readable labels
    flag_labels = {
        'historic_period': 'References a Historic Period',
        'location': 'References a Location',
        'traditional_customs_practices': 'References Customs or Practices',
        'query_language_equals_metadata_language': 'Query Lang Equals Metadata',
        'goes_beyond_metadata_using_image_contents': 'Goes Beyond Metadata',
        'references_individual': 'References an Individual',
    }
    
    for query in queries:
        # Ensure flags are booleans
        flags = {flag_labels[flag]: bool(query.get(flag, False)) for flag in flags_list}
        query_id = query['query_id']
    
        # For each method, collect rank and flags
        for rank_key, method_name in zip(method_keys, method_names):
            ranks = query.get(rank_key)
            if ranks and len(ranks) > 0:
                worst_rank = ranks[-1]  # Worst case rank (highest number)
                if worst_rank > 0:
                    data = {
                        'method': method_name,
                        'rank': worst_rank,
                        'query_id': query_id
                    }
                    data.update(flags)
                    data_list.append(data)
    
    # Convert to DataFrame
    mrr_df = pd.DataFrame(data_list)
    
    # Get top 5 metadata languages
    metadata_language_counts = pd.Series([q['metadata_language'] for q in queries]).value_counts()
    top_5_metadata_languages = metadata_language_counts.nlargest(5).index.tolist()
    
    # Function to map languages to top 5 or 'Other'
    def map_language(lang):
        return lang if lang in top_5_metadata_languages else 'Other'
    
    # Add language columns to mrr_df
    mrr_df['metadata_language'] = mrr_df['query_id'].map({q['query_id']: q['metadata_language'] for q in queries})
    mrr_df['query_language'] = mrr_df['query_id'].map({q['query_id']: q['query_language'] for q in queries})
    mrr_df['language_group'] = mrr_df['metadata_language'].apply(map_language)
    mrr_df['query_lang_equals_metadata_lang'] = mrr_df['query_language'] == mrr_df['metadata_language']
    
    # Store additional stats
    stats['mrr_df'] = mrr_df
    stats['flag_labels'] = flag_labels
    stats['top_5_metadata_languages'] = top_5_metadata_languages
    
    # Area calculations have been removed as per your request
    
    return stats


def plot_overall_cumulative(stats, queries):
    mrr_df = stats['mrr_df']
    total_queries = len(mrr_df['query_id'].unique())
    cumulative_data = []
    for method_name, method_group in mrr_df.groupby("method"):
        sorted_ranks = np.sort(method_group["rank"])
        cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
        cumulative_data.extend({
            "method": method_name,
            "rank": rank,
            "cumulative_count": cum_count
        } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

    cumulative_df = pd.DataFrame(cumulative_data)
    cumulative_df = cumulative_df.groupby(['method', 'rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
    total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
    cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cumulative_df, x="rank", y="cumulative_percent", hue="method", alpha=0.7, markers=True)

    plt.xscale("log")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Cumulative Percentage (\%)")
    plt.title(f"Cumulative Percentages by Method (All Queries, Total = {total_queries})")
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

    # Display area data
    overall_areas = stats['overall_areas']
    print("\nArea under the curve (log-scaled x-axis) for Overall Cumulative:")
    for method, area in overall_areas.items():
        print(f"Method: {method}, Area: {area:.4f}")

    # Bar plot of areas
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(overall_areas.keys()), y=list(overall_areas.values()), palette='pastel')
    plt.title('Area under the Curve by Method (Overall)')
    plt.ylabel('Area')
    plt.tight_layout()
    plt.show()


def plot_per_flag_cumulative(stats, queries, flags_group, figure_title, method_names):
    mrr_df = stats['mrr_df']
    flag_labels = stats['flag_labels']

    fig, axes = plt.subplots(len(flags_group), 2, figsize=(15, 5 * len(flags_group)))
    axes = axes.ravel()

    for i, flag in enumerate(flags_group):
        for j, flag_value in enumerate([True, False]):
            ax = axes[i*2 + j]
            flag_label = flag_labels[flag]
            df_filtered = mrr_df[mrr_df[flag_label] == flag_value]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{flag_label} = {flag_value}')
                ax.set_axis_off()
                continue

            cumulative_data = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_data.extend({
                    "method": method_name,
                    "rank": rank,
                    "cumulative_count": cum_count
                } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

            cumulative_df = pd.DataFrame(cumulative_data)
            cumulative_df = cumulative_df.groupby(['method', 'rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
            total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
            cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df[cumulative_df['method'] == method_name]
                ax.plot(method_data['rank'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            ax.set_title(f'{flag_label} = {flag_value} (Total = {total_queries})')
            ax.set_xscale('log')
            ax.set_xlabel('Rank (log scale)')
            ax.set_ylabel('Cumulative Percentage (\%)')
            ax.legend()

    fig.suptitle(figure_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # # Display area data
    # flag_area_df = stats['flag_area_df']
    # print(f"\nArea under the curve (log-scaled x-axis) for {figure_title}:")
    # print(flag_area_df.to_string(index=False))

    # # Bar plot - combined for all flags
    # area_df_melted = flag_area_df.melt(
    #     id_vars=['Flag', 'Value', 'Total Queries'],
    #     value_vars=method_names,
    #     var_name='Method',
    #     value_name='Area'
    # )
    # area_df_melted['Condition'] = area_df_melted['Flag'] + ' = ' + area_df_melted['Value'].astype(str)

    # # Combined plot for all flags
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=area_df_melted, x='Condition', y='Area', hue='Method', ci=None)
    # plt.title('Area under the Curve by Method for All Flags')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()


def plot_language_cumulative(stats, queries, method_names):
    mrr_df = stats['mrr_df']
    top_languages = stats['top_5_metadata_languages'] + ['Other']
    language_groups = top_languages

    num_languages = len(language_groups)
    subplots_per_fig = 6  # 12 subplots over two figures
    for fig_idx in range(2):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.ravel()
        start_idx = fig_idx * subplots_per_fig
        end_idx = start_idx + subplots_per_fig
        subplot_idx = 0
        for idx in range(start_idx, end_idx):
            if idx >= num_languages * 2:
                break
            lang_idx = idx // 2
            equals_idx = idx % 2
            lang = language_groups[lang_idx]
            equals = [True, False][equals_idx]

            ax = axes[subplot_idx]
            df_filtered = mrr_df[
                (mrr_df['language_group'] == lang) &
                (mrr_df['query_lang_equals_metadata_lang'] == equals)
            ]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                equals_str = "=" if equals else r"$\neq$"
                ax.set_title(f'Language: {lang}\nTotal={total_queries}\nQuery Lang {equals_str} Metadata Lang')
                ax.set_axis_off()
                subplot_idx += 1
                continue

            cumulative_data = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_data.extend({
                    "method": method_name,
                    "rank": rank,
                    "cumulative_count": cum_count
                } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

            cumulative_df = pd.DataFrame(cumulative_data)
            cumulative_df = cumulative_df.groupby(['method', 'rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
            total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
            cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df[cumulative_df['method'] == method_name]
                ax.plot(method_data['rank'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            equals_str = "=" if equals else r"$\neq$"
            ax.set_title(f'Language: {lang} (Total={total_queries})\nQuery Lang {equals_str} Metadata Lang')
            ax.set_xscale('log')
            ax.set_xlabel('Rank (log scale)')
            ax.set_ylabel('Cumulative Percentage (\%)')
            ax.legend()

            subplot_idx +=1

        fig.suptitle(f'Cumulative Percentages by Method (Languages {fig_idx+1})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # # Display area data
    # language_area_df = stats['language_area_df']
    # print("\nArea under the curve (log-scaled x-axis) for Language Cumulative:")
    # print(language_area_df.to_string(index=False))

    # # Bar plots
    # area_df_melted = language_area_df.melt(
    #     id_vars=['Language', 'Query Lang Equals Metadata Lang', 'Total Queries'],
    #     value_vars=method_names,
    #     var_name='Method',
    #     value_name='Area'
    # )
    # area_df_melted['Condition'] = area_df_melted.apply(
    #     lambda row: f"{row['Language']} - {'Equal' if row['Query Lang Equals Metadata Lang'] else 'Not Equal'}",
    #     axis=1
    # )

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=area_df_melted, x='Condition', y='Area', hue='Method', ci=None)
    # plt.title('Area under the Curve by Method for Language Cumulative')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()


def plot_language_cumulative(stats, queries, method_names):
    mrr_df = stats['mrr_df']
    top_languages = stats['top_5_metadata_languages'] + ['Other']
    language_groups = top_languages

    num_languages = len(language_groups)
    subplots_per_fig = 6  # 12 subplots over two figures
    for fig_idx in range(2):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.ravel()
        start_idx = fig_idx * subplots_per_fig
        end_idx = start_idx + subplots_per_fig
        subplot_idx = 0
        for idx in range(start_idx, end_idx):
            if idx >= num_languages * 2:
                break
            lang_idx = idx // 2
            equals_idx = idx % 2
            lang = language_groups[lang_idx]
            equals = [True, False][equals_idx]

            ax = axes[subplot_idx]
            df_filtered = mrr_df[
                (mrr_df['language_group'] == lang) &
                (mrr_df['query_lang_equals_metadata_lang'] == equals)
            ]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                equals_str = "=" if equals else r"$\neq$"
                ax.set_title(f'Language: {lang}\nTotal={total_queries}\nQuery Lang {equals_str} Metadata Lang')
                ax.set_axis_off()
                subplot_idx += 1
                continue

            cumulative_data = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_data.extend({
                    "method": method_name,
                    "rank": rank,
                    "cumulative_count": cum_count
                } for rank, cum_count in zip(sorted_ranks, cumulative_counts))

            cumulative_df = pd.DataFrame(cumulative_data)
            cumulative_df = cumulative_df.groupby(['method', 'rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')
            total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
            cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

            for method_name in df_filtered['method'].unique():
                method_data = cumulative_df[cumulative_df['method'] == method_name]
                ax.plot(method_data['rank'], method_data['cumulative_percent'], label=method_name, alpha=0.7)

            equals_str = "=" if equals else r"$\neq$"
            ax.set_title(f'Language: {lang} (Total={total_queries})\nQuery Lang {equals_str} Metadata Lang')
            ax.set_xscale('log')
            ax.set_xlabel('Rank (log scale)')
            ax.set_ylabel('Cumulative Percentage (\%)')
            ax.legend()

            subplot_idx +=1

        fig.suptitle(f'Cumulative Percentages by Method (Languages {fig_idx+1})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # # Display area data
    # language_area_df = stats['language_area_df']
    # print("\nArea under the curve (log-scaled x-axis) for Language Cumulative:")
    # print(language_area_df.to_string(index=False))

    # # Bar plots
    # area_df_melted = language_area_df.melt(
    #     id_vars=['Language', 'Query Lang Equals Metadata Lang', 'Total Queries'],
    #     value_vars=method_names,
    #     var_name='Method',
    #     value_name='Area'
    # )
    # area_df_melted['Condition'] = area_df_melted.apply(
    #     lambda row: f"{row['Language']} - {'Equal' if row['Query Lang Equals Metadata Lang'] else 'Not Equal'}",
    #     axis=1
    # )

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=area_df_melted, x='Condition', y='Area', hue='Method', ci=None)
    # plt.title('Area under the Curve by Method for Language Cumulative')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()


def plot_language_confusion_matrix(stats, queries):
    # Get the language confusion matrix
    language_confusion = stats['language_confusion_collapsed']

    # Sum over query languages to get frequencies of metadata languages
    metadata_lang_totals = language_confusion.sum(axis=0)
    
    # Sort the metadata languages by frequency in descending order
    sorted_metadata_langs = metadata_lang_totals.sort_values(ascending=False).index.tolist()

    # Reorder both rows and columns of the confusion matrix based on sorted metadata languages
    language_confusion_sorted = language_confusion.reindex(index=sorted_metadata_langs, columns=sorted_metadata_langs)

    plt.figure(figsize=(12, 10))
    sns.heatmap(language_confusion_sorted, annot=True, cmap='coolwarm', fmt='g')
    plt.title('Language Confusion Matrix (Top 20 Languages + Other)')
    plt.xlabel('Metadata Language')
    plt.ylabel('Query Language')
    plt.tight_layout()
    plt.show()

def plot_queries_per_image(stats, queries):
    image_query_counts = list(stats['image_query_count'].values())
    total_images = len(stats['image_query_count'])

    plt.figure(figsize=(8, 6))
    plt.hist(image_query_counts, bins=20, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of Queries per Image (Total Images = {total_images})')
    plt.xlabel('Number of Queries')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.show()

def plot_flag_distribution(stats, queries):
    flag_labels = stats['flag_labels']
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

def plot_top_referenced_images(stats, queries):
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

def plot_cooccurrence_matrix(stats, queries):
    flag_labels = stats['flag_labels']
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
    retrievers = [
        ClipRetriever(schema_name="baseline", host=host),
        CaptionDenseRetriever(schema_name="no-metadata", host=host),
        CaptionDenseRetriever(schema_name="with-metadata", host=host),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.9, caption_dense_weight=0.1),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.8, caption_dense_weight=0.2),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.7, caption_dense_weight=0.3),
        ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.6, caption_dense_weight=0.4)#,
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.5, caption_dense_weight=0.5),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.4, caption_dense_weight=0.6),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.3, caption_dense_weight=0.7),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.2, caption_dense_weight=0.8),
        # ClipDenseCaptionFusionRetriever(schema_name="with-metadata", host=host, clip_weight=0.1, caption_dense_weight=0.9)
    ]

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

    # Visualizations
    plot_overall_cumulative(stats, queries)

    # Now, split the flags into two groups of three
    flags_group1 = [
        'historic_period',
        'location',
        'traditional_customs_practices'
    ]
    flags_group2 = [
        'query_language_equals_metadata_language',
        'goes_beyond_metadata_using_image_contents',
        'references_individual'
    ]
    method_names = [retriever.get_method_name() for retriever in retrievers]


    plot_per_flag_cumulative(stats, queries, flags_group1, 'Cumulative Percentages by Method (Flags Group 1)', method_names)
    plot_per_flag_cumulative(stats, queries, flags_group2, 'Cumulative Percentages by Method (Flags Group 2)', method_names)

    # Plot cumulative subplots for top 5 languages and 'Other'
    plot_language_cumulative(stats, queries, method_names)

    # Other plotting functions
    plot_language_confusion_matrix(stats, queries)
    plot_queries_per_image(stats, queries)
    plot_flag_distribution(stats, queries)
    plot_top_referenced_images(stats, queries)
    plot_cooccurrence_matrix(stats, queries)

    # Close the database connection
    cur.close()
    conn.close()


# Run the main function
if __name__ == '__main__':
    main()

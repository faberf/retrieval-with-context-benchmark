import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import psycopg2  # Import psycopg2 for PostgreSQL connection

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

# Function to load queries from the PostgreSQL database
def load_queries():
    # Query to fetch data from the query and image tables
    sql = """
    SELECT q.query_id, q.query_text, q.historic_period, q.location, q.traditional_customs_practices,
        q.goes_beyond_metadata_using_image_contents, q.references_individual,
        q.query_language, q.query_language_equals_metadata_language,
        q.image_id::text, i.metadata_language
    FROM image_query_schema.query q
    JOIN image_query_schema.image i ON q.image_id = i.image_id
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
            'image_id': row[9],  # UUID converted to string
            'metadata_language': row[10]
        }
        queries.append(query)

    # Build a mapping from query_id to query dictionary for easy lookup
    query_dict = {q['query_id']: q for q in queries}

    # Define result tables and corresponding rank keys, including retrieval schemas
    result_tables = [
        ('clip_results', 'clip_rank', 'baseline'),
        ('caption_dense_results', 'nometadata_rank', 'no-metadata'),
        ('caption_dense_results', 'metadata_rank', 'with-metadata'),
        ('clip_dense_caption_fusion_results', 'fusion_rank', 'with-metadata')
    ]

    for table_name, rank_key, retrieval_schema in result_tables:
        sql = f"""
        SELECT query_id, rank
        FROM image_query_schema.{table_name}
        WHERE retrieval_schema = %s
        """
        cur.execute(sql, (retrieval_schema,))
        results = cur.fetchall()
        for query_id, rank in results:
            # Get the corresponding query dictionary
            if query_id in query_dict:
                query = query_dict[query_id]
                query[rank_key] = rank  # Directly assign the rank

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

# Function to calculate area under the curve on log-scaled x-axis
def calculate_log_area(cumulative_df):
    # Ensure the DataFrame is sorted by rank
    cumulative_df = cumulative_df.sort_values('rank')
    cumulative_df = cumulative_df.reset_index(drop=True)
    
    # Extract rank and cumulative percentage values
    x_values = cumulative_df['rank'].values
    y_values = cumulative_df['cumulative_percent'].values
    
    # Compute log10 of x values
    log_x_values = np.log10(x_values)
    
    # Initialize total area
    total_area = 0.0
    
    # Iterate over intervals
    for i in range(len(x_values) - 1):
        # Compute difference in log10(x)
        delta_log_x = log_x_values[i+1] - log_x_values[i]
        
        # Compute average of y-values
        avg_y = (y_values[i] + y_values[i+1]) / 2.0
        
        # Compute area of trapezoid
        area = avg_y * delta_log_x
        
        # Add to total area
        total_area += area
    
    return total_area

# Function to compute stats and prepare data for plotting
def compute_stats(queries, image_ids):
    stats = {}
    
    # Filter queries to only include those with ranks from all methods
    filtered_queries = [
        query for query in queries
        if query.get('clip_rank') and query.get('metadata_rank') and query.get('nometadata_rank')
        and query['clip_rank'] > 0 and query['metadata_rank'] > 0 and query['nometadata_rank'] > 0
    ]
    
    print(f"Total queries with ranks from all methods: {len(filtered_queries)}")
    
    # Use filtered_queries for stats computation
    queries = filtered_queries
    
    # Queries per image
    image_query_count = Counter([query['image_id'] for query in queries])
    
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
    language_confusion = pd.crosstab([q['query_language'] for q in queries], 
                                     [q['metadata_language'] for q in queries])
    
    # Get top 20 languages
    top_languages = language_confusion.sum().nlargest(20).index.tolist()
    
    # Collapse non-top languages into "Other"
    collapsed_query_languages = [collapse_language(q['query_language'], top_languages) for q in queries]
    collapsed_metadata_languages = [collapse_language(q['metadata_language'], top_languages) for q in queries]
    language_confusion_collapsed = pd.crosstab(collapsed_query_languages, collapsed_metadata_languages)

    # Collect reciprocal ranks
    clip_reciprocal_ranks = []
    metadata_reciprocal_ranks = []
    nometadata_reciprocal_ranks = []

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
        if query.get('clip_rank') is not None:
            clip_reciprocal_ranks.append(1.0 / query['clip_rank'])
        else:
            clip_reciprocal_ranks.append(0.0)

        if query.get('metadata_rank') is not None:
            metadata_reciprocal_ranks.append(1.0 / query['metadata_rank'])
        else:
            metadata_reciprocal_ranks.append(0.0)

        if query.get('nometadata_rank') is not None:
            nometadata_reciprocal_ranks.append(1.0 / query['nometadata_rank'])
        else:
            nometadata_reciprocal_ranks.append(0.0)
    
    # Store all stats
    stats['image_query_count'] = image_query_count
    stats['flag_counts'] = flag_counts
    stats['co_occurrence_matrix'] = co_occurrence_matrix
    stats['flag_names'] = flag_names
    stats['language_confusion_collapsed'] = language_confusion_collapsed
    stats['clip_reciprocal_ranks'] = clip_reciprocal_ranks
    stats['metadata_reciprocal_ranks'] = metadata_reciprocal_ranks
    stats['nometadata_reciprocal_ranks'] = nometadata_reciprocal_ranks

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
        for method_key, method_name in [
            ('clip_rank', 'CLIP'),
            ('metadata_rank', 'With Metadata'),
            ('nometadata_rank', 'Without Metadata')
        ]:
            rank = query.get(method_key)
            if rank is not None and rank > 0:
                data = {
                    'method': method_name,
                    'rank': rank,
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

    # Compute area under the curve for overall cumulative plot
    cumulative_data = []
    for method_name, method_group in mrr_df.groupby("method"):
        sorted_ranks = np.sort(method_group["rank"])
        cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
        cumulative_percent = (cumulative_counts / len(sorted_ranks)) * 100
        cumulative_data.extend({
            "method": method_name,
            "rank": rank,
            "cumulative_percent": cum_percent
        } for rank, cum_percent in zip(sorted_ranks, cumulative_percent))

    cumulative_df = pd.DataFrame(cumulative_data)
    overall_areas = {}
    for method_name in cumulative_df['method'].unique():
        method_df = cumulative_df[cumulative_df['method'] == method_name]
        area = calculate_log_area(method_df)
        overall_areas[method_name] = area
    stats['overall_areas'] = overall_areas

    # Compute areas under the curve for per-flag cumulative plots
    flag_area_data = []
    for flag in flags_list:
        flag_label = flag_labels[flag]
        for flag_value in [True, False]:
            df_filtered = mrr_df[mrr_df[flag_label] == flag_value]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                continue

            cumulative_data = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_percent = (cumulative_counts / len(sorted_ranks)) * 100
                cumulative_data.extend({
                    "method": method_name,
                    "rank": rank,
                    "cumulative_percent": cum_percent
                } for rank, cum_percent in zip(sorted_ranks, cumulative_percent))

            cumulative_df = pd.DataFrame(cumulative_data)
            areas = {}
            for method_name in cumulative_df['method'].unique():
                method_df = cumulative_df[cumulative_df['method'] == method_name]
                area = calculate_log_area(method_df)
                areas[method_name] = area

            area_entry = {
                'Flag': flag_label,
                'Value': flag_value,
                'Total Queries': total_queries
            }
            area_entry.update(areas)
            flag_area_data.append(area_entry)
    flag_area_df = pd.DataFrame(flag_area_data)
    stats['flag_area_df'] = flag_area_df

    # Compute areas under the curve for language cumulative plots
    language_area_data = []
    for lang in top_5_metadata_languages + ['Other']:
        for equals in [True, False]:
            df_filtered = mrr_df[
                (mrr_df['language_group'] == lang) &
                (mrr_df['query_lang_equals_metadata_lang'] == equals)
            ]
            total_queries = len(df_filtered['query_id'].unique())

            if df_filtered.empty:
                continue

            cumulative_data = []
            for method_name, method_group in df_filtered.groupby("method"):
                sorted_ranks = np.sort(method_group["rank"])
                cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
                cumulative_percent = (cumulative_counts / len(sorted_ranks)) * 100
                cumulative_data.extend({
                    "method": method_name,
                    "rank": rank,
                    "cumulative_percent": cum_percent
                } for rank, cum_percent in zip(sorted_ranks, cumulative_percent))

            cumulative_df = pd.DataFrame(cumulative_data)
            areas = {}
            for method_name in cumulative_df['method'].unique():
                method_df = cumulative_df[cumulative_df['method'] == method_name]
                area = calculate_log_area(method_df)
                areas[method_name] = area

            area_entry = {
                'Language': lang,
                'Query Lang Equals Metadata Lang': equals,
                'Total Queries': total_queries
            }
            area_entry.update(areas)
            language_area_data.append(area_entry)
    language_area_df = pd.DataFrame(language_area_data)
    stats['language_area_df'] = language_area_df

    return stats

# Plotting functions (unchanged)
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

def plot_per_flag_cumulative(stats, queries, flags_group, figure_title):
    mrr_df = stats['mrr_df']
    flag_labels = stats['flag_labels']

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
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

    # Display area data
    flag_area_df = stats['flag_area_df']
    print(f"\nArea under the curve (log-scaled x-axis) for {figure_title}:")
    print(flag_area_df.to_string(index=False))

    # Bar plot - combined for all flags
    area_df_melted = flag_area_df.melt(id_vars=['Flag', 'Value', 'Total Queries'], value_vars=['CLIP', 'With Metadata', 'Without Metadata'], var_name='Method', value_name='Area')
    area_df_melted['Condition'] = area_df_melted['Flag'] + ' = ' + area_df_melted['Value'].astype(str)

    # Combined plot for all flags
    plt.figure(figsize=(12, 6))
    sns.barplot(data=area_df_melted, x='Condition', y='Area', hue='Method', ci=None)
    plt.title(f'Area under the Curve by Method for All Flags')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_language_cumulative(stats, queries):
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

    # Display area data
    language_area_df = stats['language_area_df']
    print("\nArea under the curve (log-scaled x-axis) for Language Cumulative:")
    print(language_area_df.to_string(index=False))

    # Bar plots
    area_df_melted = language_area_df.melt(id_vars=['Language', 'Query Lang Equals Metadata Lang', 'Total Queries'], value_vars=['CLIP', 'With Metadata', 'Without Metadata'], var_name='Method', value_name='Area')
    area_df_melted['Condition'] = area_df_melted.apply(lambda row: f"{row['Language']} - {'Equal' if row['Query Lang Equals Metadata Lang'] else 'Not Equal'}", axis=1)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=area_df_melted, x='Condition', y='Area', hue='Method', ci=None)
    plt.title('Area under the Curve by Method for Language Cumulative')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

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
    # Load the queries from the database
    queries = load_queries()
    
    print(len(queries), "queries loaded.")
    
    if not queries:
        print("No queries loaded.")
        return
    
    # Get image IDs from the database
    image_ids = get_image_ids()
    
    # Compute stats
    stats = compute_stats(queries, image_ids)
    
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
    
    plot_per_flag_cumulative(stats, queries, flags_group1, 'Cumulative Percentages by Method (Flags Group 1)')
    plot_per_flag_cumulative(stats, queries, flags_group2, 'Cumulative Percentages by Method (Flags Group 2)')
    
    # Plot cumulative subplots for top 5 languages and 'Other'
    plot_language_cumulative(stats, queries)
    
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

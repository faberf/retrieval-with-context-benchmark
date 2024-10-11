import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

# Function to load queries from JSON file
def load_queries(output_queries_file):
    try:
        with open(output_queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        return queries
    except Exception as e:
        print(f"Error loading queries: {e}")
        return None

# Function to get image ids from the image directory
def get_image_ids_from_directory(images_folder):
    image_ids = set()
    for filename in os.listdir(images_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_id = os.path.splitext(filename)[0]  # Get filename without extension
            image_ids.add(image_id)
    return image_ids

# Function to collapse languages outside the top 20 into "other"
def collapse_language(lang, top_languages):
    return lang if lang in top_languages else 'other'

# Function to compute basic stats and move cumulative logic inside (updated)
def compute_stats(queries, image_ids):
    stats = {}
    
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
    
    # Collapse non-top languages into "other"
    collapsed_query_languages = [collapse_language(q['query_language'], top_languages) for q in queries]
    collapsed_metadata_languages = [collapse_language(q['metadata_language'], top_languages) for q in queries]
    language_confusion_collapsed = pd.crosstab(collapsed_query_languages, collapsed_metadata_languages)

    # Collect reciprocal ranks
    clip_reciprocal_ranks = []
    metadata_reciprocal_ranks = []
    nometadata_reciprocal_ranks = []

    # Initialize dictionary to hold cumulative rank data per flag
    cumulative_data_flags = {flag: {'clip': [], 'metadata': [], 'nometadata': []} for flag in flag_counts.keys()}

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
            for flag in flag_counts.keys():
                if query.get(flag, False):
                    cumulative_data_flags[flag]['clip'].append(1.0 / query['clip_rank'])
        else:
            clip_reciprocal_ranks.append(0.0)

        if query.get('metadata_rank') is not None:
            metadata_reciprocal_ranks.append(1.0 / query['metadata_rank'])
            for flag in flag_counts.keys():
                if query.get(flag, False):
                    cumulative_data_flags[flag]['metadata'].append(1.0 / query['metadata_rank'])
        else:
            metadata_reciprocal_ranks.append(0.0)

        if query.get('nometadata_rank') is not None:
            nometadata_reciprocal_ranks.append(1.0 / query['nometadata_rank'])
            for flag in flag_counts.keys():
                if query.get(flag, False):
                    cumulative_data_flags[flag]['nometadata'].append(1.0 / query['nometadata_rank'])
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
    stats['cumulative_data_flags'] = cumulative_data_flags
    
    return stats

# Function to visualize stats (updated)
def visualize_stats(stats):
    # Flip the display order of the cumulative percentage plot and others
    
    # Assuming the `stats` object has the reciprocal ranks data
    # Prepare the data for visualization

    # Create a DataFrame with ranks for each method
    ranks_data = []
    for rank, method in zip(stats['clip_reciprocal_ranks'], ['clip'] * len(stats['clip_reciprocal_ranks'])):
        if rank > 0:  # Avoid adding ranks with 0, since they represent no rank
            ranks_data.append({"method": method, "rank": 1 / rank})

    for rank, method in zip(stats['metadata_reciprocal_ranks'], ['metadata'] * len(stats['metadata_reciprocal_ranks'])):
        if rank > 0:
            ranks_data.append({"method": method, "rank": 1 / rank})

    for rank, method in zip(stats['nometadata_reciprocal_ranks'], ['nometadata'] * len(stats['nometadata_reciprocal_ranks'])):
        if rank > 0:
            ranks_data.append({"method": method, "rank": 1 / rank})

    # Convert to DataFrame
    mrr_df = pd.DataFrame(ranks_data)

    # Compute cumulative data without bins
    cumulative_data = []
    for method_name, method_group in mrr_df.groupby("method"):
        sorted_ranks = np.sort(method_group["rank"])
        cumulative_counts = np.arange(1, len(sorted_ranks) + 1)
        cumulative_data.extend({"method": method_name, "rank": rank, "cumulative_count": cum_count} 
                            for rank, cum_count in zip(sorted_ranks, cumulative_counts))

    # Create a DataFrame from cumulative data
    cumulative_df = pd.DataFrame(cumulative_data)

    # Group by method and rank to sum cumulative counts
    cumulative_df = cumulative_df.groupby(['method', 'rank']).size().groupby(level=0).cumsum().reset_index(name='cumulative_count')

    # Normalize cumulative counts to percentages
    total_counts = cumulative_df.groupby('method')['cumulative_count'].transform('max')
    cumulative_df['cumulative_percent'] = (cumulative_df['cumulative_count'] / total_counts) * 100

    # Create the figure and gridspec for two plots
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

    # Upper plot for the cumulative percentage plot
    ax0 = plt.subplot(gs[0])
    sns.lineplot(data=cumulative_df, x="rank", y="cumulative_percent", hue="method", ax=ax0, alpha=0.7, markers=True)

    ax0.set_xscale("log")  # Set log scale for rank
    ax0.set_xlabel("Rank (log scale)")
    ax0.set_ylabel("Cumulative Percentage")
    ax0.set_title("Cumulative Percentages by Method")

    # Remove the legend from the upper plot
    ax0.legend_.remove()

    # Lower plot for the legend
    ax1 = plt.subplot(gs[1])
    ax1.axis('off')  # Turn off the axis
    legend = ax1.legend(*ax0.get_legend_handles_labels(), loc='center', title="Method", bbox_to_anchor=(0.5, 0.5))
    ax1.add_artist(legend)

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.show()
    
    # Metadata tag subplots (cumulative plot filtered by each tag and its opposite)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, (flag, cumulative_data) in enumerate(stats['cumulative_data_flags'].items()):
        for method, data in cumulative_data.items():
            if data:
                sns.lineplot(data=np.array(data), ax=axes[i], label=method, alpha=0.7)
        axes[i].set_title(f'Cumulative for {flag}')
        axes[i].set_xscale('log')
        axes[i].set_xlabel('Rank (log scale)')
        axes[i].set_ylabel('Cumulative Count')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    # Plot the language confusion matrix with top 20 languages
    plt.figure(figsize=(12, 10))
    sns.heatmap(stats['language_confusion_collapsed'], annot=True, cmap='coolwarm', fmt='g')
    plt.title('Language Confusion Matrix (Top 20 Languages + Other)')
    plt.xlabel('Metadata Language')
    plt.ylabel('Query Language')
    plt.show()

    # Now plot the remaining stats as before (Queries per image, Flags distribution, etc.)
    # Number of queries per image (Histogram)
    image_query_counts = list(stats['image_query_count'].values())
    
    plt.figure(figsize=(8, 6))
    plt.hist(image_query_counts, bins=20, color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Queries per Image')
    plt.xlabel('Number of Queries')
    plt.ylabel('Number of Images')
    plt.show()

    # Distribution of flags (Bar chart)
    plt.figure(figsize=(8, 6))
    plt.bar(stats['flag_counts'].keys(), stats['flag_counts'].values(), color='green')
    plt.title('Counts of Metadata Flags in Queries')
    plt.xlabel('Metadata Flag')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # Top referenced images (Bar chart)
    top_images = stats['image_query_count'].most_common(10)
    image_ids, query_counts = zip(*top_images)
    
    plt.figure(figsize=(10, 6))
    plt.barh(image_ids, query_counts, color='orange')
    plt.title('Top 10 Referenced Images')
    plt.xlabel('Number of Queries')
    plt.ylabel('Image ID')
    plt.gca().invert_yaxis()
    plt.show()

    # Visualize the co-occurrence matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(stats['co_occurrence_matrix'], annot=True, cmap='Blues', xticklabels=stats['flag_names'], yticklabels=stats['flag_names'])
    plt.title('Co-occurrence Matrix of Metadata Flags')
    plt.xlabel('Metadata Flags')
    plt.ylabel('Metadata Flags')
    plt.show()


# Main function to execute the process
def main():
    images_folder = '../../benchmark/completed_images_with_categories'  # Update this with the actual folder path
    output_queries_file = 'augmented_output.json'
    
    # Load the queries
    queries = load_queries(output_queries_file)
    
    print(len(queries), "queries loaded.")
    
    if queries is None:
        print("No queries loaded.")
        return
    
    # Get image IDs from the images folder
    image_ids = get_image_ids_from_directory(images_folder)
    
    # Compute stats
    stats = compute_stats(queries, image_ids)
    
    # Visualize stats
    visualize_stats(stats)

# Run the main function
if __name__ == '__main__':
    main()

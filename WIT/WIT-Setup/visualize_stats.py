import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Function to compute basic stats
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
    
    for query in queries:
        flags = [query.get(flag, False) for flag in flag_counts]
        
        # Update flag counts
        for i, flag in enumerate(flag_counts):
            if flags[i]:
                flag_counts[flag] += 1
        
        # Update co-occurrence matrix
        for i in range(len(flags)):
            for j in range(i, len(flags)):
                if flags[i] and flags[j]:
                    co_occurrence_matrix[i, j] += 1
                    if i != j:
                        co_occurrence_matrix[j, i] += 1  # Ensure symmetry
    
    stats['image_query_count'] = image_query_count
    stats['flag_counts'] = flag_counts
    stats['co_occurrence_matrix'] = co_occurrence_matrix
    stats['flag_names'] = flag_names
    
    return stats

# Function to visualize stats
def visualize_stats(stats):
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
    output_queries_file = 'output_queries_improved.json'
    
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

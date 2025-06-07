import json
from collections import Counter
import statistics

def analyze_spaces_data(filepath="spaces_data.json"):
    """
    Performs initial analysis on the Hugging Face Spaces data.
    """
    try:
        with open(filepath, 'r') as f:
            spaces_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return

    # Total Spaces
    total_spaces = len(spaces_data)
    print(f"Total Spaces: {total_spaces}")
    print("-" * 30)

    if not spaces_data:
        print("No data to analyze.")
        return

    # SDK Analysis
    sdk_counts = Counter()
    for space in spaces_data:
        sdk = space.get('sdk')
        if sdk:  # Only count if sdk is not None or empty
            sdk_counts[sdk] += 1

    print("SDK Frequencies:")
    if sdk_counts:
        for sdk, count in sdk_counts.most_common():
            print(f"  - {sdk}: {count}")
    else:
        print("  No SDK information available.")
    print("-" * 30)

    # Tag Analysis
    all_tags = []
    for space in spaces_data:
        tags = space.get('tags', [])
        if tags: # Ensure tags is not None
            all_tags.extend(tags)

    tag_counts = Counter(all_tags)
    print("Top 20 Most Common Tags:")
    if tag_counts:
        for tag, count in tag_counts.most_common(20):
            print(f"  - {tag}: {count}")
    else:
        print("  No tags available.")
    print("-" * 30)

    # Likes Analysis
    likes_list = []
    for space in spaces_data:
        likes = space.get('likes')
        if isinstance(likes, int): # Ensure likes is an integer
            likes_list.append(likes)
        else:
            likes_list.append(0) # Default to 0 if missing or not an int

    print("Likes Analysis:")
    if likes_list:
        min_likes = min(likes_list)
        max_likes = max(likes_list)
        avg_likes = statistics.mean(likes_list)
        median_likes = statistics.median(likes_list)

        print(f"  - Minimum Likes: {min_likes}")
        print(f"  - Maximum Likes: {max_likes}")
        print(f"  - Average Likes: {avg_likes:.2f}")
        print(f"  - Median Likes: {median_likes}")
    else:
        print("  No likes data available to analyze.")
    print("-" * 30)

if __name__ == "__main__":
    analyze_spaces_data()

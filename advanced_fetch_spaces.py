import json
import pandas as pd
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse
import pickle

def ensure_nltk_data_path():
    user_nltk_data = os.path.expanduser('~/nltk_data')
    if user_nltk_data not in nltk.data.path:
        nltk.data.path.append(user_nltk_data)

def download_nltk_stopwords_resource():
    ensure_nltk_data_path()
    # print("\nChecking and downloading NLTK 'stopwords' resource...") # Quieter
    try:
        stopwords.words('english')
    except LookupError:
        print("NLTK 'stopwords' resource not found. Downloading...")
        nltk.download('stopwords', quiet=False)
        try:
            stopwords.words('english')
            print("NLTK 'stopwords' resource downloaded and accessible.")
        except Exception as e:
            print(f"Error accessing 'stopwords' after download: {e}.")
    except Exception as e:
        print(f"An error occurred while checking for stopwords: {e}. Attempting download.")
        try:
            nltk.download('stopwords', quiet=False)
            stopwords.words('english')
            print("NLTK 'stopwords' resource downloaded and accessible after error.")
        except Exception as e_dl:
            print(f"Critical error: Could not make 'stopwords' available: {e_dl}")

def preprocess_text_simple(text_data, space_id_fallback):
    current_text_to_process = ""
    if isinstance(text_data, str) and text_data.strip():
        current_text_to_process = text_data
    elif isinstance(space_id_fallback, str):
        current_text_to_process = space_id_fallback.split('/')[-1].replace('-', ' ')
    else:
        return []

    lower_text = current_text_to_process.lower()
    no_punct_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    tokens = no_punct_text.split()

    try:
        eng_stopwords = stopwords.words('english')
    except LookupError:
        print("Error: Stopwords not available in preprocess_text_simple. Returning raw tokens.")
        return [word for word in tokens if word.strip() and word.isalpha()]

    processed_tokens = [
        word for word in tokens if word not in eng_stopwords and word.strip() and word.isalpha()
    ]
    return processed_tokens

def get_top_n_similar(space_id, cosine_sim_matrix, df, n=5):
    if space_id not in df['id'].values:
        print(f"Error: Space ID {space_id} not found in DataFrame for similarity lookup.")
        return []
    space_idx = df.index[df['id'] == space_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim_matrix[space_idx]))
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_scores_indices = sorted_sim_scores[1:n+1]
    result = []
    for i, score in top_scores_indices:
        if i < len(df):
            result.append((df['id'].iloc[i], score))
        else:
            print(f"Warning: Index {i} out of bounds for DataFrame of length {len(df)}.")
    return result

def main_processing_pipeline():
    download_nltk_stopwords_resource()

    input_directory = "/app/"
    input_filename = "categorized_spaces_data.json" # This is the input for this script run
    absolute_input_path = os.path.join(input_directory, input_filename)

    if not os.path.exists(absolute_input_path):
        print(f"Error: Input file {absolute_input_path} was not found.")
        return

    try:
        df = pd.read_json(absolute_input_path, orient='records')
    except Exception as e:
        print(f"Error loading JSON into DataFrame: {e}")
        return

    df['processed_text'] = df.apply(
        lambda row: preprocess_text_simple(
            str(row['cardData']) if pd.notna(row['cardData']) else '',
            row['id']
        ),
        axis=1
    )
    df['joined_processed_text'] = df['processed_text'].apply(lambda tokens: ' '.join(tokens))

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))

    try:
        tfidf_matrix = vectorizer.fit_transform(df['joined_processed_text'])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)

        # --- Exploration part (kept as per instruction) ---
        print("\n--- Exploring Top N Similar Spaces (Sample) ---")
        example_indices = [0, 10, 20]
        if len(df) < 21:
            example_indices = [i for i in range(min(len(df), 3))]
        for idx_val in example_indices:
            if idx_val < len(df):
                example_space_id = df['id'].iloc[idx_val]
                print(f"\nTop N similar spaces for: {example_space_id}")
                original_space_details = df[df['id'] == example_space_id].iloc[0]
                print(f"  Category: {original_space_details['category']}")
                print(f"  Processed Text (first 15): {original_space_details['processed_text'][:15]}...")
                top_similar = get_top_n_similar(example_space_id, cosine_sim_matrix, df, n=5)
                for sim_id, score in top_similar:
                    print(f"  - {sim_id} (Score: {score:.4f})")
                    # similar_space_details = df[df['id'] == sim_id].iloc[0] # Could add more details if needed
            else:
                print(f"\nSkipping example index {idx_val} as DataFrame is too small.")
        print("\n--- End of Exploration ---")
        # --- End of Exploration part ---

        # --- Saving artifacts ---
        print("\n--- Saving Processed Data and Models ---")

        # Save the DataFrame
        df_output_filename = "/app/processed_spaces_with_text.json"
        df.to_json(df_output_filename, orient='records', indent=4)
        print(f"Processed DataFrame saved to {df_output_filename}")

        # Save the TF-IDF matrix
        tfidf_matrix_filename = "/app/tfidf_matrix.npz"
        scipy.sparse.save_npz(tfidf_matrix_filename, tfidf_matrix)
        print(f"TF-IDF matrix saved to {tfidf_matrix_filename}")

        # Save the Cosine Similarity matrix
        cosine_sim_matrix_filename = "/app/cosine_similarity_matrix.npy"
        np.save(cosine_sim_matrix_filename, cosine_sim_matrix)
        print(f"Cosine similarity matrix saved to {cosine_sim_matrix_filename}")

        # Save the TfidfVectorizer object
        vectorizer_filename = "/app/tfidf_vectorizer.pkl"
        with open(vectorizer_filename, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"TfidfVectorizer saved to {vectorizer_filename}")

        print("\n--- All artifacts saved successfully. ---")

    except ValueError as ve:
        print(f"ValueError during TF-IDF or Cosine Similarity: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during processing or saving: {e}")

if __name__ == "__main__":
    main_processing_pipeline()

# Categories and categorize_space function (kept for reference, not used in main_processing_pipeline directly)
categories = {
    "Voice/Audio Processing": ["speech", "audio", "voice", "tts", "asr", "speaker", "sound", "musicgeneration"],
}
def categorize_space(title_or_id_suffix, tags):
    return "Other"

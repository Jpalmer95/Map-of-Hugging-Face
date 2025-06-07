import json
import pandas as pd
from huggingface_hub import HfApi
import os

# Categories dictionary
categories = {
    "Voice/Audio Processing": ["speech", "audio", "voice", "tts", "asr", "speaker", "sound", "musicgeneration"],
    "Image Processing/Generation": ["image", "diffusion", "gan", "jpeg", "vision", "segmentation", "inpainting", "outpainting", "lora", "dreambooth", "controlnet", "photomaker", "transparent-background", "face-swap", "ip-adapter"],
    "Video Processing/Generation": ["video", "motion", "animation", "streamdiffusion"],
    "Text Processing/Generation": ["text", "nlp", "llm", "language-model", "summarization", "translation", "rag", "chat", "mistral", "llama", "gemma", "bert", "gpt", "phi", "rwkv", "bitnet"],
    "Code Generation/Assistants": ["code", "coding", "developer-tools"],
    "Machine Learning/AI Tools": ["pytorch", "tensorflow", "jax", "onnx", "tensorrt", "autotrain", "transformers", "diffusers", "peft", "accelerate", "optimum", "trl", "keras", "explainability", "synthetic-data", "model-editing"],
    "Data Science/Analytics": ["data", "csv", "json", "visualization", "dashboard", "embed", "webqna"],
    "Gaming/Simulation": ["game", "gaming"],
    "Robotics/Agents": ["robotics", "agent"],
    "Educational/Research": ["research", "education", "arxiv"],
    "Utilities/Miscellaneous": ["utils", "tool", "docker", "gradio", "streamlit", "api", "official", "community", "template", "course", "tutorial", "other", "web", "mcp-server", "mcp-client", "mcp-vision-receiver", "mcp-vision-sender", "mcp-controller", "mcp-agent", "mcp-tts", "mcp-asr", "mcp-comfyui"],
}

def categorize_space(title_or_id_suffix, tags):
    if not isinstance(tags, list):
        tags = []
    normalized_tags = [str(tag).lower() for tag in tags]
    normalized_title = str(title_or_id_suffix).lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if any(keyword in tag for tag in normalized_tags):
                return category
            if keyword in normalized_title:
                return category
    return "Other"


def fetch_process_categorize_and_save_final_data():
    hf_api = HfApi()
    fetch_limit = 200
    print(f"Fetching {fetch_limit} spaces from Hugging Face Hub...")
    spaces_generator = hf_api.list_spaces(full=True, limit=fetch_limit)

    all_spaces_data_raw = []
    count = 0
    for space_info in spaces_generator:
        space_dict = {
            'id': space_info.id,
            'author': space_info.author,
            'likes': space_info.likes,
            'tags': getattr(space_info, 'tags', []),
            'sdk': getattr(space_info, 'sdk', None),
            'cardData': getattr(space_info, 'cardData', None),
            'models': getattr(space_info, 'models', []),
            'datasets': getattr(space_info, 'datasets', [])
        }
        all_spaces_data_raw.append(space_dict)
        count += 1
        if count % 100 == 0:
            print(f"Fetched {count} spaces so far...")

    total_fetched_before_filter = len(all_spaces_data_raw)
    print(f"\nTotal number of spaces fetched before filtering: {total_fetched_before_filter}")

    filtered_spaces_intermediate = [
        space for space in all_spaces_data_raw if space.get('likes', 0) >= 2
    ]
    total_fetched_after_filter = len(filtered_spaces_intermediate)
    print(f"Total number of spaces after filtering (likes >= 2): {total_fetched_after_filter}")

    processed_for_df = []
    for space in filtered_spaces_intermediate:
        processed_space = space.copy()
        card_data_obj = processed_space['cardData']
        if card_data_obj is not None:
            if hasattr(card_data_obj, 'text'):
                processed_space['cardData'] = card_data_obj.text
            elif isinstance(card_data_obj, dict) or isinstance(card_data_obj, list):
                try:
                    processed_space['cardData'] = json.dumps(card_data_obj)
                except TypeError:
                    processed_space['cardData'] = str(card_data_obj)
            else:
                processed_space['cardData'] = str(card_data_obj)
        else:
            processed_space['cardData'] = ""
        processed_for_df.append(processed_space)

    if processed_for_df:
        df = pd.DataFrame(processed_for_df)

        df['category'] = df.apply(lambda x: categorize_space(x['id'].split('/')[-1], x['tags']), axis=1)

        print("\nCategory Distribution:") # Ensure this header is printed
        print(df['category'].value_counts())

        print("\nDataFrame Head with Category:")
        pd.set_option('display.max_colwidth', 50)
        print(df[['id', 'author', 'likes', 'sdk', 'tags', 'category']].head())

        print("\nDataFrame Info (with category):")
        df.info(verbose=True)

        # Define the NEW output filename for categorized data
        output_directory = "/app/"
        categorized_output_filename = "categorized_spaces_data.json"
        absolute_categorized_path = os.path.join(output_directory, categorized_output_filename)

        try:
            df.to_json(absolute_categorized_path, orient='records', indent=4)
            print(f"\nCategorized data successfully saved to {absolute_categorized_path}")

            if os.path.exists(absolute_categorized_path):
                print(f"Verification: File '{absolute_categorized_path}' exists.")
                stat_info = os.stat(absolute_categorized_path)
                print(f"Verification: File size: {stat_info.st_size} bytes.")
            else:
                print(f"Verification ERROR: File '{absolute_categorized_path}' does NOT exist after saving.")
        except Exception as e:
            print(f"\nError saving DataFrame to JSON: {e}")
    else:
        print("\nNo spaces matched. DataFrame not created, nothing to save or categorize.")

if __name__ == "__main__":
    fetch_process_categorize_and_save_final_data()

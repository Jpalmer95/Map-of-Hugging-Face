import json
from huggingface_hub import HfApi, SpaceInfo

# Initialize HfApi
hf_api = HfApi()

# Fetch 500 spaces with detailed information
print("Fetching spaces from Hugging Face Hub...")
spaces_generator = hf_api.list_spaces(full=True, limit=500)

spaces_data = []
for space in spaces_generator:
    # Convert SpaceInfo object to a dictionary to ensure serializability
    # We can select specific fields or convert the whole object
    # For simplicity, converting known serializable fields.
    # If SpaceInfo has many fields and some might not be serializable,
    # it's better to explicitly pick fields.
    space_dict = {
        "id": space.id,
        "author": space.author,
        "sha": space.sha,
        "lastModified": space.lastModified,
        "private": space.private,
        "disabled": space.disabled,
        "gated": space.gated,
        "likes": space.likes,
        "sdk": space.sdk,
        "runtime": space.runtime.to_dict() if space.runtime else None, # RuntimeInfo also needs to be dict
        "models": space.models,
        "datasets": space.datasets,
        "tags": space.tags,
        "cardData": space.cardData,
        "siblings": [{"rfilename": s.rfilename} for s in space.siblings] if space.siblings else None, # Simplified siblings
        # Add other fields as necessary, ensuring they are serializable
    }
    spaces_data.append(space_dict)

print(f"Fetched data for {len(spaces_data)} spaces.")

# Save the data to a JSON file
output_filename = "spaces_data.json"
print(f"Saving data to {output_filename}...")
with open(output_filename, "w") as f:
    json.dump(spaces_data, f, indent=4, default=str) # Use default=str for datetime etc.

print(f"Successfully saved data to {output_filename}")

import os
import requests

# ===========================
# CONFIGURATION (EDIT THIS)
# ===========================

# Your Hugging Face token (required for private or large files)
HF_API_TOKEN = "hf_xxx_your_token_here"

# Direct download URL for mistral-7b-openorca.Q4_0.gguf
# Example (replace with your model URL):
HF_URL = "https://huggingface.co/owner/repo-name/resolve/main/mistral-7b-openorca.Q4_0.gguf"

# Where to save the model on your system
MODEL_SAVE_PATH = r"D:\models\mistral-7b-openorca.Q4_0.gguf"
# Example:
# MODEL_SAVE_PATH = r"C:\Users\YourName\Models\mistral-7b-openorca.Q4_0.gguf"

# ===========================
# DO NOT EDIT BELOW
# ===========================

def download_file(url, output_path, token):
    """Download large files from Hugging Face with authentication."""

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1MB chunks

        print(f"Downloading to: {output_path}")
        print(f"File size: {round(total_size / (1024 * 1024), 2)} MB")

        with open(output_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)

        print("\n✅ Download completed successfully!")
        print("Saved at:", output_path)

    else:
        print("❌ Download failed!")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


if __name__ == "__main__":
    print("🚀 Starting model download...")
    download_file(HF_URL, MODEL_SAVE_PATH, HF_API_TOKEN)

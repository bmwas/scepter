import os
import subprocess
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv(dotenv_path=".env")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_TOKEN not found in .env or environment!")

def curl_download(hf_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://huggingface.co/scepter-studio/ACE-0.6B-512px/resolve/main/{hf_path}"
    print(f"Downloading {url} -> {local_path}")
    subprocess.run([
        "curl", "-L", "-H", f"Authorization: Bearer {HF_TOKEN}",
        "-o", local_path, url
    ], check=True)

# Manually enumerated files for text_encoder and tokenizer
txtenc_dir = "models/text_encoder/t5-v1_1-xxl"
tokenizer_dir = "models/tokenizer/t5-v1_1-xxl"

# Update these lists based on actual files in the repo directories
text_encoder_files = [
    f"{txtenc_dir}/config.json",
    f"{txtenc_dir}/pytorch_model-00001-of-00005.bin",
    f"{txtenc_dir}/pytorch_model-00002-of-00005.bin",
    f"{txtenc_dir}/pytorch_model-00003-of-00005.bin",
    f"{txtenc_dir}/pytorch_model-00004-of-00005.bin",
    f"{txtenc_dir}/pytorch_model-00005-of-00005.bin",
    f"{txtenc_dir}/pytorch_model.bin.index.json",
]

tokenizer_files = [
    f"{tokenizer_dir}/special_tokens_map.json",
    f"{tokenizer_dir}/spiece.model",
    f"{tokenizer_dir}/tokenizer.json",
    f"{tokenizer_dir}/tokenizer_config.json",
]

# Download single files
curl_download("models/dit/ace_0.6b_512px.pth", "./cache/cache_data/models/dit/ace_0.6b_512px.pth")
curl_download("models/vae/vae.bin", "./cache/cache_data/models/vae/vae.bin")

# Download all files in the text encoder and tokenizer directories
for f in text_encoder_files:
    curl_download(f, f"./cache/cache_data/{f}")

for f in tokenizer_files:
    curl_download(f, f"./cache/cache_data/{f}")

print("All required model files have been downloaded.")

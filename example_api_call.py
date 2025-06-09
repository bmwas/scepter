import requests
import base64

API_URL = "http://localhost:8000/generate"

# You can change the prompt and other parameters as needed
data = {
    "prompt": "a futuristic city at sunset"
}

response = requests.post(API_URL, json=data)
response.raise_for_status()

result = response.json()
img_b64 = result["image_base64"]

# Decode and save the image
with open("generated_image.png", "wb") as f:
    f.write(base64.b64decode(img_b64))

print("Image saved as generated_image.png")

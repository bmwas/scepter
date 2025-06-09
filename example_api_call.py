import requests
import base64
import io
from PIL import Image

def main():
    # FastAPI server endpoint - use localhost for local testing or IP for cross-container
    # If running client outside the Docker container where server is running,
    # you may need to replace localhost with the container's IP or hostname
    API_URL = "http://localhost:8000/generate"
    
    # Uncomment and modify this line if running from another container or remote machine
    # API_URL = "http://<container_ip_or_hostname>:8000/generate"
    
    # Parameters for image generation - matching the PromptRequest model in app.py
    data = {
        "prompt": "a futuristic city at sunset",
        "negative_prompt": "",
        "output_height": 512,
        "output_width": 512,
        "sample_steps": 20,
        "guide_scale": 4.5,
        "guide_rescale": 0.5,
        "seed": 2024  # Fixed seed for reproducibility
    }
    
    print(f"Sending request to {API_URL} with prompt: '{data['prompt']}'...")
    
    try:
        # Call the FastAPI endpoint
        response = requests.post(API_URL, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Process the response
        result = response.json()
        if "image_base64" not in result:
            print(f"Error: Response does not contain 'image_base64'. Response: {result}")
            return
            
        img_b64 = result["image_base64"]
        
        # Decode base64 to image
        image_data = base64.b64decode(img_b64)
        
        # Save the image
        output_path = "generated_image.png"
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"Image successfully generated and saved to {output_path}")
        
        # Also display the image if in an environment that supports it
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"Image dimensions: {image.width}x{image.height}")
        except Exception as img_e:
            print(f"Note: Could not open the image with PIL: {img_e}")
        
        # Also save the base64 for reference
        with open("generated_image_base64.txt", "w") as f:
            f.write(img_b64)
        print("Base64 version of the image saved to generated_image_base64.txt")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}\nResponse content: {response.text if 'response' in locals() else 'No response'}")
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running with: python3 app.py")
    except Exception as e:
        print(f"Other error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

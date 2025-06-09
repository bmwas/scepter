import requests
import base64
import io
from PIL import Image

def main():
    # FastAPI server endpoint - use localhost for local testing or IP for cross-container
    # If running client outside the Docker container where server is running,
    # you may need to replace localhost with the container's IP or hostname
    MODE = "generate"  # or "editing"
    BASE_URL = "http://64.247.196.8:8000"  # Update host if different
    API_URL = f"{BASE_URL}/{MODE}"
    
    # Uncomment and modify this line if running from another container or remote machine
    # API_URL = "http://<container_ip_or_hostname>:8000/generate"
    
    # -------------- Choose between generation or editing -------------- #
    if MODE == "generate":
        data = {
            "prompt": "a child with a red hat and blue shirt",
            "negative_prompt": "",
            "output_height": 512,
            "output_width": 512,
            "sample_steps": 20,
            "guide_scale": 4.5,
            "guide_rescale": 0.5,
            "seed": 2024
        }
    else:  # MODE == "editing"
        # Load a local image to edit
        src_path = "source.png"  # Provide any local image path
        with open(src_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Optional mask path (white = edit area, black = keep). Comment out if not needed
        mask_b64 = None
        # mask_path = "mask.png"
        # with open(mask_path, "rb") as f:
        #     mask_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        data = {
            "image_base64": img_b64,
            "mask_base64": mask_b64,
            "prompt": "make the hat green",
            "task": "inpainting",
            "negative_prompt": "",
            "output_height": 512,
            "output_width": 512,
            "sample_steps": 20,
            "guide_scale": 4.5,
            "guide_rescale": 0.5,
            "seed": 42
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

import requests
import base64
import io
import sys
from pathlib import Path
from typing import Optional
from PIL import Image

import requests
import base64
import io
import sys
from pathlib import Path
from typing import Optional
from PIL import Image
import argparse

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

EXAMPLES = f"""
EXAMPLES:
  # Text-to-Image (generation mode):
  python example_api_call.py --mode generate --prompt "a child with a red hat and blue shirt"

  # Image Editing (edit mode, no mask):
  python example_api_call.py --mode editing --source_img myphoto.jpg --edit_prompt "make the hat green"

  # Image Editing (edit mode, with mask):
  python example_api_call.py --mode editing --source_img myphoto.jpg --mask_img mymask.png --edit_prompt "make the hat green"
"""

def encode_image_to_base64(path: str, pil_mode: Optional[str] = "RGB") -> str:
    """Load any supported image, convert to given mode, encode as PNG base64."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file '{path}' does not exist")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{p.suffix}'. Supported: {', '.join(SUPPORTED_EXTS)}")
    with Image.open(p) as im:
        if pil_mode:
            im = im.convert(pil_mode)
        buf = io.BytesIO()
        im.save(buf, format="PNG")  # unify format for API
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser(
        description="Scepter API Example Client: Text-to-Image and Image Editing",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLES
    )
    parser.add_argument('--mode', choices=['generate', 'editing'], required=True, help='Operation mode: generate or editing')
    parser.add_argument('--prompt', type=str, help='Prompt for text-to-image (generation mode)')
    parser.add_argument('--source_img', type=str, help='Path to image to edit (editing mode)')
    parser.add_argument('--mask_img', type=str, default=None, help='Path to mask image (editing mode, optional)')
    parser.add_argument('--edit_prompt', type=str, default=None, help='Edit instruction (editing mode)')
    parser.add_argument('--task', type=str, default="", help='Task type: empty string (default), "inpainting", etc.')
    parser.add_argument('--api_url', type=str, default=None, help='Override API base URL (default: http://64.247.196.8:8000)')
    parser.add_argument('--output', type=str, default='generated_image.png', help='Output file name')
    args = parser.parse_args()

    BASE_URL = args.api_url or "http://64.247.196.41:8000"
    MODE = args.mode
    API_URL = f"{BASE_URL}/{MODE}"

    print(f"\n--- Scepter API Client ---")
    print(f"Mode: {MODE}")
    print(f"Endpoint: {API_URL}")

    if MODE == "generate":
        if not args.prompt:
            print("\n[ERROR] --prompt is required for generate mode!\n")
            parser.print_help()
            return
        data = {
            "prompt": args.prompt,
            "negative_prompt": "",
            "output_height": 512,
            "output_width": 512,
            "sample_steps": 20,
            "guide_scale": 4.5,
            "guide_rescale": 0.5,
            "seed": 2024
        }
    else:  # editing
        if not args.source_img:
            print("\n[ERROR] --source_img is required for editing mode!\n")
            parser.print_help()
            return
        if not args.edit_prompt:
            print("\n[ERROR] --edit_prompt is required for editing mode!\n")
            parser.print_help()
            return
        try:
            img_b64 = encode_image_to_base64(args.source_img, "RGB")
        except Exception as e:
            print(f"[ERROR] Could not load source image: {e}")
            return
        mask_b64 = None
        if args.mask_img:
            try:
                mask_b64 = encode_image_to_base64(args.mask_img, "L")
            except Exception as e:
                print(f"[ERROR] Could not load mask image: {e}")
                return
        data = {
            "image_base64": img_b64,
            "mask_base64": mask_b64,
            "prompt": args.edit_prompt,
            "task": args.task,  # Empty string by default, allows ACE to determine best task
            "negative_prompt": "",
            "output_height": 512,
            "output_width": 512,
            "sample_steps": 50,
            "guide_scale": 2.5,
            "guide_rescale": 0.5,
            "seed": 42
        }
    print(f"\nPayload being sent:\n{data}\n")
    print(f"Sending request to {API_URL} ...")
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        result = response.json()
        if "image_base64" not in result:
            print(f"Error: Response does not contain 'image_base64'. Response: {result}")
            return
        img_b64 = result["image_base64"]
        image_data = base64.b64decode(img_b64)
        output_path = args.output
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"\nImage successfully generated and saved to {output_path}")
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"Image dimensions: {image.width}x{image.height}")
        except Exception as img_e:
            print(f"Note: Could not open the image with PIL: {img_e}")
        with open("generated_image_base64.txt", "w") as f:
            f.write(img_b64)
        print("Base64 version of the image saved to generated_image_base64.txt\n")
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

if __name__ == "__main__":
    main()

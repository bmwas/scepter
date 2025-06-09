import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import os

# Scepter imports
from scepter.modules.utils.config import Config
from scepter.modules.inference.ace_inference import ACEInference

# Path to your config file
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'scepter', 'methods', 'edit', 'wandb_dit_ace_0.6b_512.yaml'
)

# Load config and model at startup
cfg = Config(cfg_file=CONFIG_PATH)
inference = ACEInference()
inference.init_from_cfg(cfg)

app = FastAPI(title="Scepter Text-to-Image API")

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ''
    output_height: int = 512
    output_width: int = 512
    sample_steps: int = 20
    guide_scale: float = 4.5
    guide_rescale: float = 0.5
    seed: int = -1

@app.post("/generate")
def generate_image(req: PromptRequest):
    try:
        # Call inference (returns list of PIL Images)
        images = inference(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            output_height=req.output_height,
            output_width=req.output_width,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            guide_rescale=req.guide_rescale,
            seed=req.seed
        )
        if not images or not isinstance(images[0], Image.Image):
            raise RuntimeError("Inference did not return a valid image.")
        img = images[0]
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"image_base64": img_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

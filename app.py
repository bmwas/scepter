import io
import base64
import time
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
import os
import logging
from typing import List, Optional

# Set up logging
logger = logging.getLogger("scepter-api")
logger.setLevel(logging.DEBUG)

# Create console handler with a specific format
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Also log to file
fh = logging.FileHandler('scepter_api.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Scepter imports
from scepter.modules.utils.config import Config
from scepter.modules.inference.ace_inference import ACEInference

# Start time for tracking server uptime
start_server_time = time.time()

# Path to your config file
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'scepter', 'methods', 'edit', 'wandb_dit_ace_0.6b_512.yaml'
)

# Function to properly set up model function_info
def initialize_model_function_info(model_dict, function_name, dtype='float16'):
    """Initialize function_info for a model to ensure get_function_info works properly"""
    if model_dict is None:
        logger.error(f"Cannot initialize function_info for None model")
        return None
    
    if 'function_info' not in model_dict:
        logger.info(f"Adding function_info dictionary to model")
        model_dict['function_info'] = {}
    
    model_dict['function_info'][function_name] = {'dtype': dtype}
    logger.debug(f"Set function_info for {function_name} with dtype {dtype}")
    return model_dict

# Load config and model at startup
logger.info(f"Loading config from {CONFIG_PATH}")
try:
    logger.info(f"Loading config from {CONFIG_PATH}")
    cfg = Config(load=True, cfg_file=CONFIG_PATH)
    logger.info("Config loaded successfully")
    
    # Add NAME attribute to config (required by ACEInference)
    logger.info("Adding NAME attribute to config")
    cfg.NAME = "ACEInference"
    
    # Extract model sections from config
    logger.info("Extracting model sections from config")
    
    if cfg.have("SOLVER") and cfg.SOLVER.have("MODEL"):
        logger.debug(f"Found MODEL in SOLVER with {len(cfg.SOLVER.MODEL)} entries")
        
        # Set up MODEL config
        logger.info("Setting up MODEL Config object")
        cfg.MODEL = Config(load=False)
        
        # Set up model subsections
        logger.info("Setting up DIFFUSION_MODEL Config object")
        cfg.MODEL.DIFFUSION_MODEL = Config(load=False)
        logger.info("Setting up FIRST_STAGE_MODEL Config object")
        cfg.MODEL.FIRST_STAGE_MODEL = Config(load=False)
        logger.info("Setting up COND_STAGE_MODEL Config object")
        cfg.MODEL.COND_STAGE_MODEL = Config(load=False)
        
        # Copy all MODEL attributes from SOLVER.MODEL
        for key, value in cfg.SOLVER.MODEL.items():
            if key == "DIFFUSION_MODEL":
                for dkey, dvalue in cfg.SOLVER.MODEL.DIFFUSION_MODEL.items():
                    cfg.MODEL.DIFFUSION_MODEL[dkey] = dvalue
            elif key == "FIRST_STAGE_MODEL":
                for fkey, fvalue in cfg.SOLVER.MODEL.FIRST_STAGE_MODEL.items():
                    cfg.MODEL.FIRST_STAGE_MODEL[fkey] = fvalue
            elif key == "COND_STAGE_MODEL":
                for ckey, cvalue in cfg.SOLVER.MODEL.COND_STAGE_MODEL.items():
                    cfg.MODEL.COND_STAGE_MODEL[ckey] = cvalue
            else:
                cfg.MODEL[key] = value
        
        logger.info("MODEL sections successfully mapped to config attributes")
    else:
        logger.error("Required MODEL section not found in config")
        raise ValueError("Config missing required MODEL section")

    # Initialize ACEInference
    logger.info("Initializing ACEInference with config...")
    start_init = time.time()
    inference = ACEInference()
    inference.init_from_cfg(cfg)  
    
    # Ensure function_info is properly initialized for each model
    logger.info("Setting up function_info for models...")
    if inference.first_stage_model:
        inference.first_stage_model = initialize_model_function_info(inference.first_stage_model, 'encode', 'float16')
        inference.first_stage_model = initialize_model_function_info(inference.first_stage_model, 'decode', 'float16')
    # For the text encoder (cond_stage_model) we need the tokenizer-aware entrypoint
    # so that raw text is properly tokenized before reaching the transformer.
    if inference.cond_stage_model:
        inference.cond_stage_model = initialize_model_function_info(
            inference.cond_stage_model, 'encode_list_of_list', 'float16')
    if inference.diffusion_model:
        inference.diffusion_model = initialize_model_function_info(inference.diffusion_model, 'forward', 'float16')
    
    # Force early model loading to avoid issues during the first request
    logger.info("Pre-loading all models to ensure they're ready...")
    try:
        inference.dynamic_load(inference.first_stage_model, 'first_stage_model')
        inference.dynamic_load(inference.cond_stage_model, 'cond_stage_model')
        inference.dynamic_load(inference.diffusion_model, 'diffusion_model')
        logger.info("All models pre-loaded successfully")
    except Exception as model_load_error:
        logger.error(f"Error during model pre-loading: {str(model_load_error)}")
        logger.error(traceback.format_exc())
    
    logger.info(f"ACEInference initialization successful (took {time.time() - start_init:.2f} seconds)")
    logger.info("Setting up FastAPI application")

except Exception as e:
    # Log any exceptions that occur during initialization
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise e

logger.info("Setting up FastAPI application")
app = FastAPI(title="Scepter Text-to-Image API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ''
    output_height: int = 512
    output_width: int = 512
    sample_steps: int = 20
    guide_scale: float = 4.5
    guide_rescale: float = 0.5
    seed: int = -1

# Request model for image editing
class EditRequest(BaseModel):
    image_base64: str  # Source image in base64 PNG/JPEG
    mask_base64: Optional[str] = None  # Optional mask (white areas will be edited)
    prompt: str  # Edit instruction
    task: str = ''  # e.g., 'inpainting' (left blank will default inside inference)
    negative_prompt: str = ''
    output_height: int = 512
    output_width: int = 512
    sample_steps: int = 20
    guide_scale: float = 4.5
    guide_rescale: float = 0.5
    seed: int = -1

# Function to prepare prompt inputs based on ACE model's expected formats
def format_ace_inputs(prompt, negative_prompt=None):
    """Format inputs for the ACE model based on its requirements.
    
    Based on ACE model's __call__ method:
    - Prompt must be a list of strings (batch style)
    - Negative prompt must be a simple string: "negative prompt text"
    
    Args:
        prompt: User-provided prompt (string or list)
        negative_prompt: User-provided negative prompt (string or list)
        
    Returns:
        Tuple of (formatted_prompt, formatted_negative_prompt) ready for model
    """
    # Provide prompt as list of strings (batch style)
    if isinstance(prompt, list):
        # Flatten nested lists and cast to string
        formatted_prompt = [p[0] if isinstance(p, list) else str(p) for p in prompt]
    else:
        formatted_prompt = [str(prompt)]

    # Negative prompt is always a string
    if negative_prompt is None:
        formatted_neg_prompt = ""
    elif isinstance(negative_prompt, list):
        formatted_neg_prompt = str(negative_prompt[0]) if len(negative_prompt) > 0 else ""
    else:
        formatted_neg_prompt = str(negative_prompt)

    logger.info(f"[format_ace_inputs] formatted_prompt type: {type(formatted_prompt)}, value: {formatted_prompt}")
    logger.info(f"[format_ace_inputs] formatted_neg_prompt type: {type(formatted_neg_prompt)}, value: {formatted_neg_prompt}")
    return formatted_prompt, formatted_neg_prompt


@app.post("/generate")
def generate_image(req: PromptRequest):
    request_id = f"req_{int(time.time() * 1000)}"  # Simple unique ID for request tracking
    logger.info(f"[{request_id}] New image generation request with prompt: '{req.prompt[:50]}...'")
    logger.debug(f"[{request_id}] Full request parameters: {req}")
    
    start_time = time.time()
    try:
        # Explicitly load models before inference
        logger.info(f"[{request_id}] Pre-loading required models for inference")
        if hasattr(inference, 'first_stage_model') and inference.first_stage_model is not None:
            inference.dynamic_load(inference.first_stage_model, 'first_stage_model')
        if hasattr(inference, 'cond_stage_model') and inference.cond_stage_model is not None:
            inference.dynamic_load(inference.cond_stage_model, 'cond_stage_model')
        if hasattr(inference, 'diffusion_model') and inference.diffusion_model is not None:
            inference.dynamic_load(inference.diffusion_model, 'diffusion_model')
        
        # Call inference (returns list of PIL Images)
        logger.info(f"[{request_id}] Running inference with sample steps: {req.sample_steps}, seed: {req.seed}")
        
        # Format prompts correctly for the ACE model
        formatted_prompt, formatted_neg_prompt = format_ace_inputs(req.prompt, req.negative_prompt)
        logger.info(f"[{request_id}] Using formatted prompt: {formatted_prompt}")
        if formatted_neg_prompt:
            logger.info(f"[{request_id}] Using negative prompt: '{formatted_neg_prompt}'")
        inference_start = time.time()
        images = inference(
            prompt=formatted_prompt,  # list of strings as expected by ACEInference
            negative_prompt=formatted_neg_prompt,  # string
            output_height=req.output_height,
            output_width=req.output_width,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            guide_rescale=req.guide_rescale,
            seed=req.seed
        )
        inference_time = time.time() - inference_start
        logger.info(f"[{request_id}] Inference completed in {inference_time:.2f} seconds")
        
        if not images:
            logger.error(f"[{request_id}] Inference returned empty image list")
            raise RuntimeError("Inference did not return any images.")
        if not isinstance(images[0], Image.Image):
            logger.error(f"[{request_id}] Inference returned non-image object: {type(images[0])}")
            raise RuntimeError("Inference did not return a valid image.")
        
        img = images[0]
        logger.info(f"[{request_id}] Image successfully generated with size {img.width}x{img.height}")
        # Convert PIL image to base64
        try:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] Request completed successfully in {total_time:.2f} seconds")
            return {"image_base64": img_b64, "processing_time": total_time}
        except Exception as e:
            logger.error(f"[{request_id}] Error converting image to base64: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
    except Exception as e:
        # Handle any errors gracefully
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Error during image generation: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "processing_time": total_time}

# ---------------------------- Image Editing Endpoint ---------------------------- #

@app.post("/editing")
def edit_image(req: EditRequest):
    request_id = f"edit_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] New image editing request with prompt: '{req.prompt[:50]}...'")
    
    start_time = time.time()
    try:
        # Decode source image
        try:
            src_img = Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to decode source image: {e}")
            raise HTTPException(status_code=400, detail="Invalid source image data")
        
        # Decode mask if present, otherwise auto-create a white mask (edit everywhere)
        if req.mask_base64:
            try:
                mask_img = Image.open(io.BytesIO(base64.b64decode(req.mask_base64))).convert("L")
            except Exception as e:
                logger.error(f"[{request_id}] Failed to decode mask image: {e}")
                raise HTTPException(status_code=400, detail="Invalid mask image data")
        else:
            # Auto-create a white mask (all 255) to allow editing everywhere
            mask_img = Image.new("L", src_img.size, 255)
        
        # Add detailed diagnostic logging
        logger.info(f"[{request_id}] Image dimensions: {src_img.size}")
        logger.info(f"[{request_id}] Mask provided: {req.mask_base64 is not None}")
        if mask_img:
            logger.info(f"[{request_id}] Mask dimensions: {mask_img.size}, Histogram: {mask_img.histogram()[:10]}...")
        logger.info(f"[{request_id}] Task: '{req.task}'")
        
        # Ensure required models are loaded
        if hasattr(inference, 'first_stage_model') and inference.first_stage_model is not None:
            inference.dynamic_load(inference.first_stage_model, 'first_stage_model')
        if hasattr(inference, 'cond_stage_model') and inference.cond_stage_model is not None:
            inference.dynamic_load(inference.cond_stage_model, 'cond_stage_model')
        if hasattr(inference, 'diffusion_model') and inference.diffusion_model is not None:
            inference.dynamic_load(inference.diffusion_model, 'diffusion_model')
        
        # Prepare prompts
        formatted_prompt, formatted_neg_prompt = format_ace_inputs(req.prompt, req.negative_prompt)
        logger.info(f"[{request_id}] Formatted prompt: {formatted_prompt}")
        
        inference_start = time.time()
        images = inference(
            image=[src_img],
            mask=[mask_img] if mask_img else [None],
            prompt=formatted_prompt,
            task=[req.task] if req.task else [''],
            negative_prompt=formatted_neg_prompt,
            output_height=req.output_height,
            output_width=req.output_width,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            guide_rescale=req.guide_rescale,
            seed=req.seed
        )
        inference_time = time.time() - inference_start
        logger.info(f"[{request_id}] Inference completed in {inference_time:.2f} seconds")
        
        if not images or not isinstance(images[0], Image.Image):
            logger.error(f"[{request_id}] Inference returned no valid image")
            raise RuntimeError("Inference failed to produce an image")
        
        # Convert to base64
        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Result image dimensions: {images[0].size}")
        logger.info(f"[{request_id}] Editing request completed in {total_time:.2f} seconds")
        return {"image_base64": img_b64, "processing_time": total_time}
    
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Error during image editing: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "processing_time": total_time}

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy", 
        "model": "ACEInference", 
        "uptime": time.time() - start_server_time
    }

if __name__ == "__main__":
    logger.info("Starting Scepter Text-to-Image API server")
    start_server_time = time.time()
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())

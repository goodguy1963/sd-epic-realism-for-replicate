import os
# Set this environment variable to disable NSFW checking
os.environ["DISABLE_SAFETY_CHECKER"] = "1"
# Set PyTorch memory management environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from typing import List
import glob
import subprocess

import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils import load_image

# Split into SAMPLERS and SCHEDULERS
SAMPLERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "DPMPP_2M": DPMSolverSinglestepScheduler,  # DPM++ 2M is implemented using DPMSolverSinglestepScheduler
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "KLMS": LMSDiscreteScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}

SCHEDULER_TYPES = [
    "default",
    "karras",
    "exponential",
    "simple",
    "discrete",
    "sde-dpmsolver++",
    "v-prediction"
]

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "model")
LORA_PATH = os.path.join(BASE_PATH, "lora")

def download_weights(api_key=None):
    """Download model weights using the download script"""
    script_path = os.path.join(BASE_PATH, "script", "download-weights")
    try:
        # Write API key to file if provided
        if api_key:
            api_key_file = os.path.join(BASE_PATH, "civitai_api_key.txt")
            with open(api_key_file, 'w') as f:
                f.write(api_key)
                
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download weights: {str(e)}")

# Available models and LoRAs
AVAILABLE_MODELS = {
    "epic_realism_v4": {
        "name": "Epic Realism v4.0",
        "filename": "epicrealismNatural_v40.safetensors",
        "type": "stable-diffusion-1-5"
    },
    "perfectdeliberate": {
        "name": "PerfectDeliberate v5.0",
        "filename": "perfectdeliberate.safetensors",
        "type": "stable-diffusion-1-5"
    },
}

AVAILABLE_LORAS = {
    "none": {
        "name": "No LoRA",
        "filename": None,
        "weight": 0.0
    },
    "more_details": {
        "name": "More Details",
        "filename": "more_details.safetensors",
        "weight": 0.7
    }
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the models into memory"""
        try:
            print("Setting up pipeline...")
            
            # Check for API key file on startup
            api_key = self._read_api_key()
            if not api_key:
                print("WARNING: No Civitai API key found. You must provide an API key with the civitai_api_key parameter.")
                print("The model will not function without a valid API key.")
            
            # Initialize dictionaries for loaded models and pipelines
            self.loaded_models = {}
            self.loaded_pipes = {}
            self.loaded_loras = {}
            
            # Load main models
            for model_id, model_info in AVAILABLE_MODELS.items():
                filename = model_info["filename"]
                model_path = os.path.join(MODEL_PATH, filename)
                
                if os.path.exists(model_path):
                    print(f"Found model file for {model_info['name']}: {model_path}")
                    self.loaded_models[model_id] = model_path
                else:
                    print(f"Warning: Model file for {model_info['name']} not found at {model_path}")
            
            # Verify we have at least one model and download if needed
            if not self.loaded_models:
                print("No models found. Attempting to download...")
                download_weights()
                
                # Check again after download
                for model_id, model_info in AVAILABLE_MODELS.items():
                    filename = model_info["filename"]
                    model_path = os.path.join(MODEL_PATH, filename)
                    if os.path.exists(model_path):
                        print(f"Successfully downloaded model file for {model_info['name']}: {model_path}")
                        self.loaded_models[model_id] = model_path
            
            # Create LoRA directory if it doesn't exist
            if not os.path.exists(LORA_PATH):
                print(f"Creating LoRA directory at {LORA_PATH}")
                os.makedirs(LORA_PATH, exist_ok=True)

            # Load LoRA models
            missing_loras = False
            for lora_id, lora_info in AVAILABLE_LORAS.items():
                if lora_info["filename"] is None:
                    continue  # Skip "none" option
                
                filename = lora_info["filename"]
                if isinstance(filename, (str, bytes, os.PathLike)):
                    lora_path = os.path.join(LORA_PATH, filename)
                    if os.path.exists(lora_path):
                        print(f"Found LoRA file: {lora_path}")
                        self.loaded_loras[lora_id] = True
                    else:
                        print(f"Warning: LoRA file not found at {lora_path}")
                        missing_loras = True
            
            # Download missing LoRAs if any
            if missing_loras:
                print("Missing LoRAs detected. Attempting to download...")
                download_weights()
                
                # Check again after download
                for lora_id, lora_info in AVAILABLE_LORAS.items():
                    if lora_info["filename"] is None:
                        continue
                    
                    filename = lora_info["filename"]
                    if isinstance(filename, (str, bytes, os.PathLike)):
                        lora_path = os.path.join(LORA_PATH, filename)
                        if os.path.exists(lora_path):
                            print(f"Successfully downloaded LoRA file: {lora_path}")
                            self.loaded_loras[lora_id] = True

            # Track device info
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            print("Setup complete. Models will be loaded on first use.")
            
        except Exception as e:
            raise RuntimeError(f"Error during setup: {str(e)}")

    def _read_api_key(self):
        """Read the API key from the civitai_api_key.txt file"""
        api_key_file = os.path.join(BASE_PATH, "civitai_api_key.txt")
        api_key = None
        
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            api_key = line
                            break
            except Exception as e:
                print(f"Error reading API key file: {str(e)}")
        
        return api_key
        
    def _load_model(self, model_id, api_key=None):
        """Load a specific model if it's not already loaded"""
        if model_id in self.loaded_pipes:
            print(f"Using already loaded model: {AVAILABLE_MODELS.get(model_id, {'name': model_id})['name']}")
            return self.loaded_pipes[model_id]
        
        print(f"Loading model: {AVAILABLE_MODELS.get(model_id, {'name': model_id})['name']}")
        
        # Get the model path
        if model_id not in self.loaded_models:
            # Try to download the missing model
            print(f"Model {model_id} not found. Attempting to download...")
            download_weights(api_key=api_key)
            
            # Check if model is now available
            filename = AVAILABLE_MODELS[model_id]["filename"]
            model_path = os.path.join(MODEL_PATH, filename)
            if os.path.exists(model_path):
                self.loaded_models[model_id] = model_path
            else:
                raise ValueError(f"Model {model_id} not found and failed to download")
            
        model_path = self.loaded_models[model_id]
        
        try:
            # Load the model
            print(f"Loading from {model_path}")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Ensure safety checker is disabled
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = None
                pipe.requires_safety_checker = False
            
            pipe = pipe.to(self.device)
            
            # Create img2img pipeline
            img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device)
            
            # Create inpaint pipeline
            inpaint_pipe = StableDiffusionInpaintPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device)
            
            # Store all pipelines
            self.loaded_pipes[model_id] = {
                "txt2img": pipe,
                "img2img": img2img_pipe,
                "inpaint": inpaint_pipe
            }
            
            return self.loaded_pipes[model_id]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    def _apply_loras(self, pipeline, lora_selection, lora_weight=None):
        """Apply selected LoRAs to the pipeline"""
        if not lora_selection or lora_selection == "none":
            print("No LoRAs selected")
            return pipeline
        
        try:
            # Ensure pipeline is on CPU for LoRA loading to avoid VRAM issues
            pipeline.unet.to("cpu")
            torch.cuda.empty_cache()  # Clear CUDA cache before LoRA operations
            
            # Get LoRA info
            lora_info = AVAILABLE_LORAS[lora_selection]
            
            # Handle single or multiple LoRAs
            lora_filenames = lora_info["filename"]
            lora_weights = lora_info["weight"]
            
            if not isinstance(lora_filenames, list):
                lora_filenames = [lora_filenames]
                lora_weights = [lora_weights]
            
            # Override weight if specified
            if lora_weight is not None:
                lora_weights = [float(lora_weight)] * len(lora_filenames)
                
            print(f"Applying LoRA: {lora_info['name']}")
            
            # Reset any existing LoRA weights
            attn_procs = {}
            for name in pipeline.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipeline.unet.config.block_out_channels[block_id]
                
                attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4)
            
            pipeline.unet.set_attn_processor(attn_procs)
            
            # Apply each LoRA
            for i, filename in enumerate(lora_filenames):
                if filename is None:
                    continue
                    
                lora_path = os.path.join(LORA_PATH, filename)
                weight = lora_weights[i]
                
                if not os.path.exists(lora_path) or os.path.getsize(lora_path) < 10000:
                    print(f"LoRA file {filename} not found or too small. Attempting to download...")
                    download_weights()
                    
                    # Verify the download was successful
                    if not os.path.exists(lora_path) or os.path.getsize(lora_path) < 10000:
                        print(f"Failed to download valid LoRA file {filename}")
                        continue
                
                print(f"Applying LoRA from {filename} with weight {weight}")
                
                try:
                    # Load the safetensors file
                    print(f"Loading LoRA weights from {lora_path}")
                    state_dict = load_file(lora_path, device="cpu")
                    
                    # Apply weights to attention processors
                    for name, attn_processor in pipeline.unet.attn_processors.items():
                        if isinstance(attn_processor, LoRAAttnProcessor):
                            for weight_name in ["to_k_lora", "to_q_lora", "to_v_lora", "to_out_lora"]:
                                up_weight_name = f"{name}.{weight_name}.up.weight"
                                down_weight_name = f"{name}.{weight_name}.down.weight"
                                
                                if up_weight_name in state_dict and down_weight_name in state_dict:
                                    attn_processor.set_lora_layer(
                                        weight_name,
                                        state_dict[up_weight_name].to(self.device) * weight,
                                        state_dict[down_weight_name].to(self.device)
                                    )
                    
                    print(f"Successfully applied LoRA {filename}")
                    
                except Exception as e:
                    print(f"Error loading LoRA file {filename}: {str(e)}")
                    if os.path.exists(lora_path):
                        os.remove(lora_path)
                    print("Attempting to redownload...")
                    download_weights()
                    continue
            
            # Move pipeline back to GPU and clear CPU memory
            pipeline.unet.to(self.device)
            torch.cuda.empty_cache()
            
            return pipeline
            
        except Exception as e:
            print(f"Error in _apply_loras: {str(e)}")
            # Move pipeline back to GPU in case of error
            pipeline.unet.to(self.device)
            torch.cuda.empty_cache()
            return pipeline

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art, mutated, deformed, distorted, disfigured, body horror, watermark, text, meme, bad proportions, cropped head, out of frame, cut off, ugly, duplicate, mutilated, mutation, disgusting, bad anatomy, bad hands, three hands, three legs, bad feet, three feet",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048],
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048],
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=500,
            default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=1,
            le=20,
            default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            ge=0.0,
            le=1.0,
            default=0.8
        ),
        model_name: str = Input(
            description="Choose which model to use",
            choices=list(AVAILABLE_MODELS.keys()),
            default="epic_realism_v4"
        ),
        lora_selection: str = Input(
            description="Select a LoRA or LoRA combination to apply",
            choices=list(AVAILABLE_LORAS.keys()),
            default="none"
        ),
        lora_weight: float = Input(
            description="LoRA weight (higher values increase effect strength)",
            ge=0.5,
            le=3.0,
            default=0.8
        ),
        sampler: str = Input(
            description="Choose a sampler",
            choices=list(SAMPLERS.keys()),
            default="DPMSolverMultistep"
        ),
        scheduler_type: str = Input(
            description="Choose a scheduler type",
            choices=SCHEDULER_TYPES,
            default="default"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
        civitai_api_key: str = Input(
            description="Your Civitai API key for downloading models (REQUIRED)",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            # Validate that API key is provided
            api_key = civitai_api_key or self._read_api_key()
            if not api_key:
                raise ValueError("Civitai API key is required. Please provide it using the civitai_api_key parameter.")
                
            print("Running prediction...")
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
            
            # Handle Civitai API key
            if civitai_api_key:
                print("API key provided, saving to file for downloads")
                api_key_file = os.path.join(BASE_PATH, "civitai_api_key.txt")
                with open(api_key_file, 'w') as f:
                    f.write(civitai_api_key)
            
            # Clear CUDA cache before starting prediction
            if torch.cuda.is_available():
                print("Clearing CUDA cache before prediction")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # Load the selected model with API key if provided
            pipes = self._load_model(model_name, api_key=civitai_api_key)
            
            # Configure sampler and scheduler
            scheduler_class = SAMPLERS[sampler]
            
            # Get the base config from the current scheduler
            scheduler_config = pipes["txt2img"].scheduler.config
            
            # Apply scheduler type if not default
            if scheduler_type != "default" and hasattr(scheduler_config, "algorithm_type"):
                scheduler_config["algorithm_type"] = scheduler_type
                print(f"Using {sampler} sampler with {scheduler_type} algorithm type")
            else:
                print(f"Using {sampler} sampler with default configuration")
            
            # Create and set the scheduler for all pipelines
            for pipe_type, pipe in pipes.items():
                pipe.scheduler = scheduler_class.from_config(scheduler_config)
            
            # Create a generator on the same device as the model
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Disable model CPU offloading for all pipelines to prevent device mismatches
            for pipe_type, pipe in pipes.items():
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe._is_model_cpu_offloaded = False
                    
                # Ensure all pipeline components are on the same device
                for component in ["vae", "text_encoder", "unet", "scheduler"]:
                    if hasattr(pipe, component):
                        component_model = getattr(pipe, component)
                        if hasattr(component_model, "to") and component_model is not None:
                            component_model.to(self.device)

            # Apply LoRAs if selected
            if lora_selection and lora_selection != "none":
                # Apply to all pipeline types
                for pipe_type, pipe in pipes.items():
                    self._apply_loras(pipe, lora_selection, lora_weight)
            else:
                print("No LoRAs selected")

            if image is not None and mask is not None:
                print("Running inpainting...")
                pipe = pipes["inpaint"]
                image = Image.open(image).convert("RGB")
                mask_image = Image.open(mask).convert("RGB")
                
                # Resize mask to match image if needed
                if mask_image.size != image.size:
                    print(f"Resizing mask to match image dimensions: {image.width}x{image.height}")
                    mask_image = mask_image.resize(image.size, Image.LANCZOS)
                
                try:
                    # Enable attention slicing for memory efficiency during inpainting
                    if hasattr(pipe, "enable_attention_slicing"):
                        pipe.enable_attention_slicing()
                    
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        mask_image=mask_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=prompt_strength,
                        generator=generator,
                    ).images[0]
                except torch.cuda.OutOfMemoryError as e:
                    print("Out of memory error. Try using a smaller input image or increasing your system's GPU memory.")
                    raise RuntimeError("CUDA out of memory. Please use a smaller input image.")
                    
            elif image is not None:
                print("Running img2img...")
                pipe = pipes["img2img"]
                print(f"Loading image from: {image}")
                image = Image.open(image).convert("RGB")
                print(f"Processing image with dimensions: {image.width}x{image.height}")
                
                try:
                    # Enable attention slicing for memory efficiency during img2img
                    if hasattr(pipe, "enable_attention_slicing"):
                        pipe.enable_attention_slicing()
                    
                    # Make sure to DISABLE model offloading for img2img to prevent device mismatches
                    if hasattr(pipe, "enable_model_cpu_offload"):
                        pipe._is_model_cpu_offloaded = False
                    
                    print(f"Starting img2img pipeline...")
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=prompt_strength,
                        generator=generator,
                    ).images[0]
                    print(f"Img2img completed successfully")
                except torch.cuda.OutOfMemoryError as e:
                    print("Out of memory error. Try using a smaller input image or increasing your system's GPU memory.")
                    raise RuntimeError("CUDA out of memory. Please use a smaller input image.")
                except Exception as e:
                    print(f"Error in img2img: {str(e)}")
                    # Try a fallback approach with more memory management
                    torch.cuda.empty_cache()
                    print("Trying alternative approach...")
                    
                    # Make sure all components are explicitly on CUDA
                    for component_name in ["vae", "text_encoder", "unet", "tokenizer"]:
                        if hasattr(pipe, component_name) and getattr(pipe, component_name) is not None:
                            component = getattr(pipe, component_name)
                            if hasattr(component, "to"):
                                component = component.to(self.device)
                                setattr(pipe, component_name, component)
                    
                    # Try again
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=prompt_strength,
                        generator=generator,
                    ).images[0]
            else:
                print("Running txt2img...")
                pipe = pipes["txt2img"]
                
                # Only enable attention slicing for large resolutions
                if width >= 768 or height >= 768:
                    if hasattr(pipe, "enable_attention_slicing"):
                        print("Enabling attention slicing for high resolution generation")
                        pipe.enable_attention_slicing()
                
                # Explicitly set output_type to avoid device issues
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil",  # Force PIL output to avoid device issues
                ).images[0]

            output_path = "/tmp/output.png"
            output.save(output_path)
            
            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            return Path(output_path)
            
        except Exception as e:
            # Clear CUDA cache if there's an error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            raise RuntimeError(f"Error during inference: {str(e)}")
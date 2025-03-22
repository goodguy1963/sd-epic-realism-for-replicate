# Epic Realism Stable Diffusion

A Stable Diffusion implementation combining Epic Realism v4.0 and PerfectDeliberate v5.0 models with LoRA support.

## Features

- Epic Realism v4.0 model support
- PerfectDeliberate v5.0 model support
- LoRA integration
- txt2img, img2img, and inpainting capabilities
- Multiple samplers and scheduler types
- CUDA optimization for better performance

## Requirements

- Python 3.11+
- CUDA capable GPU
- Civitai API key

## Setup

1. Clone this repository:
```bash
git clone [your-repository-url]
cd sd-epic-realism
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `civitai_api_key.txt` file and add your Civitai API key to it.

4. The models and LoRAs will be downloaded automatically on first run.

## Usage

The model can be used through Cog or directly through Python. 

### Input Parameters

- `prompt`: Input prompt for image generation
- `negative_prompt`: Things to avoid in the generated image
- `image`: Input image for img2img or inpainting
- `mask`: Mask image for inpainting
- `width`: Output image width (128-2048)
- `height`: Output image height (128-2048)
- `num_inference_steps`: Number of denoising steps (1-500)
- `guidance_scale`: Scale for classifier-free guidance (1-20)
- `prompt_strength`: Strength of prompt when using init image (0-1)
- `model_name`: Choose between "epic_realism_v4" and "perfectdeliberate"
- `lora_selection`: Available LoRA options
- `lora_weight`: LoRA effect strength (0.5-3.0)
- `sampler`: Choice of different sampling methods
- `scheduler_type`: Type of scheduler to use
- `seed`: Random seed for reproducibility

## Models

- Epic Realism v4.0: A photorealistic model focused on natural images
- PerfectDeliberate v5.0: High-quality general purpose model

## License

[Include your license information here]
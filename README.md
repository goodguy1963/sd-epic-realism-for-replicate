# Epic Realism Stable Diffusion

A Stable Diffusion implementation combining Epic Realism v4.0 and PerfectDeliberate v5.0 models with LoRA support. This project creates a Replicate API using Claude 3.7 Agent via VS Code Insider Build.

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

## Getting Started

### Obtain Civitai API Key
1. Create an account on [Civitai](https://civitai.com)
2. Go to Settings → API Keys
3. Generate a new API key
4. Save the key in `civitai_api_key.txt` in the project root

### Running with Cog

1. Install Cog:
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

2. Build the model:
```bash
cog build
```

3. Run predictions:
```bash
cog predict -i prompt="your prompt here"
```

Note: When using Windows, it's recommended to use WSL (Windows Subsystem for Linux) as your default terminal for better compatibility.

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

## Models and LoRA

- Epic Realism Natural Sin Final SD1.5 by epinikion
- PerfectDeliberate by Desync
- "More Details" LoRA by Lykon

## License

[Include your license information here]
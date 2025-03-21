# Epic Realism Stable Diffusion Model

This is a deployment of the Epic Realism Natural v4.0 checkpoint for Stable Diffusion on Replicate. The model produces highly realistic images with enhanced detail and lighting.

## Model Description

Epic Realism Natural v4.0 is a Stable Diffusion checkpoint that specializes in generating photorealistic images. It excels at:
- Portrait photography with realistic skin textures and lighting
- Landscape and nature scenes with enhanced detail
- Realistic product and still life photography

## Inputs

- **prompt** - Text prompt describing the image you want to generate
- **negative_prompt** - Text describing what you don't want to see in the generated image
- **width** - Width of the output image (default: 512)
- **height** - Height of the output image (default: 512)
- **num_outputs** - Number of images to generate (default: 1)
- **num_inference_steps** - Number of denoising steps (default: 30)
- **guidance_scale** - Scale for classifier-free guidance (default: 7.5)
- **scheduler** - Choice of scheduler algorithm (default: DPMSolverMultistep)
- **seed** - Random seed for reproducible results (optional)

## Example Usage

```python
import replicate

output = replicate.run(
    "yourusername/epic-realism:latest",
    input={
        "prompt": "portrait of a woman with blue eyes, photorealistic, detailed, professional photography, natural light",
        "negative_prompt": "worst quality, low quality, deformed, distorted",
        "width": 512,
        "height": 768,
        "num_outputs": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "scheduler": "DPMSolverMultistep"
    }
)

print(output)
```

## Credits

Model: Epic Realism Natural v4.0 checkpoint by [Original Creator]
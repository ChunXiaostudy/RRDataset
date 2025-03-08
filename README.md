# AI Image Generation Pipeline

A high-performance image generation pipeline that leverages multiple GPUs to generate high-quality images using LLMs and Stable Diffusion models.

## Features

- Multi-GPU support with efficient task distribution
- Parallel image generation using multiple models (SD3.5 and Flux)
- LLM-powered prompt generation and enhancement
- Theme-based image organization
- 4-bit quantization for optimal GPU memory usage

## Requirements

- Python 3.8+
- CUDA-capable GPUs (optimized for 8x GPUs setup)
- 16GB+ GPU memory per card
- 32GB+ system RAM

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare configuration files:
   - Create `path.txt` with model paths:
     ```
     qwen2.5_7b:/path/to/qwen/model
     stable-diffusion-3.5-large:/path/to/sd3.5/model
     FLUX.1-dev:/path/to/flux/model
     ```
   - Create `base_prompt.json` with themes and scenes:
     ```json
     {
       "theme1": ["scene1", "scene2"],
       "theme2": ["scene1", "scene2"]
     }
     ```

## Usage

1. Run the image generation pipeline:
```bash
python batch_image_generator.py
```

The script will:
- Load themes and prompts from configuration files
- Generate enhanced prompts using Qwen LLM
- Distribute generation tasks across available GPUs
- Save generated images in the `generated_images` directory

## Project Structure

- `batch_image_generator.py`: Main script for parallel image generation
- `config.py`: Configuration management
- `image_generator.py`: Image generation pipeline implementation
- `llm_generator.py`: LLM-based prompt generation
- `path.txt`: Model path configurations
- `base_prompt.json`: Theme and scene definitions

## GPU Distribution

The system automatically distributes tasks across available GPUs:
- GPUs 0-3: SD3.5 image generation
- GPUs 4-7: Flux image generation

## Output Structure

Generated images are saved in the following format:
```
generated_images/
    ├── theme1/
    │   ├── theme1_sd35_0.png
    │   ├── theme1_flux_0.png
    │   └── ...
    └── theme2/
        ├── theme2_sd35_0.png
        ├── theme2_flux_0.png
        └── ...
```

## Notes

- The system is optimized for 8x GPU setup but can be adapted for different configurations
- Model paths should be absolute paths to ensure proper loading
- Ensure sufficient disk space for generated images

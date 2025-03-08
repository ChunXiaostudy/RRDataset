import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from typing import List
import os

class ImageGenerator:
    def __init__(self, config, gpu_id: int = 0):
        self.config = config
        self.device = f"cuda:{gpu_id}"
        
        # Load SD3 model
        self.sd3_path = config.get_model_path("stable-diffusion-3-medium")
        self.sd3 = StableDiffusionPipeline.from_pretrained(
            self.sd3_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        self.sd3.scheduler = DPMSolverMultistepScheduler.from_config(self.sd3.scheduler.config)
        
        # Load SDXL model
        self.sdxl_path = config.get_model_path("stable-diffusion-xl-base-1.0")
        self.sdxl = StableDiffusionPipeline.from_pretrained(
            self.sdxl_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        self.sdxl.scheduler = DPMSolverMultistepScheduler.from_config(self.sdxl.scheduler.config)
        
    def generate_images(self, prompt: str, theme: str, num_images: int = 2) -> List[str]:
        # Enhance prompt
        enhanced_prompt = self._enhance_prompt(prompt)
        
        output_paths = []
        # Generate one image with each model
        
        # Generate with SD3
        image_sd3 = self.sd3(
            prompt=enhanced_prompt,
            negative_prompt=self._get_negative_prompt(),
            num_inference_steps=60,
            guidance_scale=7.5
        ).images[0]
        
        # Generate with SDXL
        image_sdxl = self.sdxl(
            prompt=enhanced_prompt,
            negative_prompt=self._get_negative_prompt(),
            num_inference_steps=60,
            guidance_scale=7.5
        ).images[0]
        
        # Save images
        save_dir = os.path.join(self.config.output_base_dir, theme)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save SD3 generated image
        sd3_path = os.path.join(save_dir, f"{theme}_sd3_{len(os.listdir(save_dir))}.png")
        image_sd3.save(sd3_path)
        output_paths.append(sd3_path)
        
        # Save SDXL generated image
        sdxl_path = os.path.join(save_dir, f"{theme}_sdxl_{len(os.listdir(save_dir))}.png")
        image_sdxl.save(sdxl_path)
        output_paths.append(sdxl_path)
        
        return output_paths
    
    def _enhance_prompt(self, prompt: str) -> str:
        # Add general quality enhancement terms
        quality_terms = "high quality, photorealistic, masterpiece, highly detailed, best quality, professional photography"
        return f"{prompt}, {quality_terms}"
    
    def _get_negative_prompt(self) -> str:
        return "low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet" 
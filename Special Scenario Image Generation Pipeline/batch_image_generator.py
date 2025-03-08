import os
import json
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from typing import List, Tuple
from tqdm import tqdm
from config import Config
import multiprocessing

class SD35Generator:
    def __init__(self, config, gpu_id: int):
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"Initializing SD3.5 on GPU {gpu_id}")
        
        model_path = config.get_model_path("stable-diffusion-3.5-large")
        print(f"Loading SD3.5 from: {model_path}")
        
        # Configure 4bit quantization
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load transformer
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        
        # Load complete pipeline
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.enable_model_cpu_offload()

class FluxGenerator:
    def __init__(self, config, gpu_id: int):
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"Initializing Flux on GPU {gpu_id}")
        
        model_path = config.get_model_path("FLUX.1-dev")
        print(f"Loading Flux from: {model_path}")
        
        self.pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.enable_model_cpu_offload()

def process_sd35_batch(args: Tuple):
    gpu_id, prompts_batch = args
    torch.cuda.set_device(gpu_id)
    config = Config()
    
    try:
        generator = SD35Generator(config, gpu_id)
        
        for theme, prompt, idx in tqdm(prompts_batch, desc=f"SD3.5 GPU {gpu_id} Progress"):
            enhanced_prompt = f"{prompt}, high detail, 4k realistic photo"
            
            try:
                image = generator.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt="bad hands, unrealistic, unresonable",
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    max_sequence_length=512
                ).images[0]
                
                save_dir = os.path.join("generated_images", theme)
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f"{theme}_sd35_{idx}.png"))
                
            except Exception as e:
                print(f"Error generating SD3.5 image for theme {theme}, prompt {idx}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error initializing SD3.5 on GPU {gpu_id}: {str(e)}")
        raise

def process_flux_batch(args: Tuple):
    gpu_id, prompts_batch = args
    torch.cuda.set_device(gpu_id)
    config = Config()
    
    try:
        generator = FluxGenerator(config, gpu_id)
        
        for theme, prompt, idx in tqdm(prompts_batch, desc=f"Flux GPU {gpu_id} Progress"):
            enhanced_prompt = f"{prompt}, high detail, 4k realistic photo"
            
            try:
                image = generator.pipeline(
                    enhanced_prompt,
                    negative_prompt="bad hands, unrealistic, unresonable",
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
                
                save_dir = os.path.join("generated_images", theme)
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f"{theme}_flux_{idx}.png"))
                
            except Exception as e:
                print(f"Error generating Flux image for theme {theme}, prompt {idx}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error initializing Flux on GPU {gpu_id}: {str(e)}")
        raise

def generate_images(prompts_file: str):
    config = Config()
    
    print(f"Loading prompts from {prompts_file}")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    # Flatten all prompts
    flat_prompts = []
    for theme, prompts in all_prompts.items():
        for i, prompt in enumerate(prompts):
            flat_prompts.append((theme, prompt, i))
    
    # Calculate prompts per GPU
    total_prompts = len(flat_prompts)
    prompts_per_gpu = total_prompts // 8
    print(f"Total prompts: {total_prompts}, Prompts per GPU: {prompts_per_gpu}")
    
    # Allocate prompts for each GPU
    gpu_batches = []
    for i in range(8):
        start_idx = i * prompts_per_gpu
        end_idx = start_idx + prompts_per_gpu if i < 7 else len(flat_prompts)
        batch = flat_prompts[start_idx:end_idx]
        gpu_batches.append((i, batch))
        print(f"GPU {i}: {len(batch)} prompts")
    
    try:
        # Process all GPUs simultaneously using multiprocessing
        with multiprocessing.Pool(processes=8) as pool:
            # Create all tasks
            tasks = []
            
            # Add SD3.5 tasks (GPU 0-3)
            print("\nStarting SD3.5 processes on GPUs 0-3...")
            for i in range(4):
                task = pool.apply_async(process_sd35_batch, (gpu_batches[i],))
                tasks.append(task)
            
            # Add Flux tasks (GPU 4-7)
            print("Starting Flux processes on GPUs 4-7...")
            for i in range(4, 8):
                task = pool.apply_async(process_flux_batch, (gpu_batches[i],))
                tasks.append(task)
            
            # Wait for all tasks to complete and check results
            print("\nWaiting for all processes to complete...")
            for i, task in enumerate(tasks):
                try:
                    task.get()  # This will raise any exceptions that occurred in the process
                    print(f"Process on GPU {i} completed successfully")
                except Exception as e:
                    print(f"Process on GPU {i} failed with error: {str(e)}")
            
            print("\nAll processes completed!")
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
    finally:
        print("\nImage generation process finished")

if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Ignore error if start method is already set
    
    print("Starting image generation process...")
    generate_images("generated_prompts.json")
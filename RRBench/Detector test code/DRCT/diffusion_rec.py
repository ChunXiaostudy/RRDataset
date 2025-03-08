from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
from PIL import Image
import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
import albumentations as A
import gc
import torch.multiprocessing as mp
from functools import partial

GenImage_LIST = [
    'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
    'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
    'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan'
]
DRCT_2M_LIST = [
    'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1',
    'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0', 'sd-turbo', 'sdxl-turbo',
    'lcm-lora-sdv1-5', 'lcm-lora-sdxl',  'sd-controlnet-canny',
    'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0', 'stable-diffusion-inpainting',
    'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1',
]

def create_crop_transforms(height=224, width=224):
    aug_list = [
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=height, width=width)
    ]
    return A.Compose(aug_list)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def find_nearest_multiple(a, multiple=8):
    
    n = a // multiple
    remainder = a % multiple
    if remainder == 0:
       
        return a
    else:
        
        return (n + 1) * multiple


def pad_image_to_size(image, target_width=224, target_height=224, fill_value=255):
    

    height, width = image.shape[:2]

    if height < target_height:
        pad_height = target_height - height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
    else:
        pad_top = pad_bottom = 0

    if width < target_width:
        pad_width = target_width - width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
    else:
        pad_left = pad_right = 0

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=fill_value
    )

    return padded_image


def center_crop(image, crop_width, crop_height):
    height, width = image.shape[:2]

   
    if width > crop_width:
        start_x = (width - crop_width) // 2
        end_x = start_x + crop_width
    else:
        start_x, end_x = 0, width
    if height > crop_height:
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height
    else:
        start_y, end_y = 0, height

    
    cropped_image = image[start_y:end_y, start_x:end_x]
    if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
        cropped_image = pad_image_to_size(cropped_image, target_width=crop_width, target_height=crop_width,
                                          fill_value=255)

    return cropped_image


def stable_diffusion_inpainting(pipe, image, mask_image, prompt, steps=50, height=512, width=512,
                                seed=2023, guidance_scale=7.5):
    set_seed(int(seed))
    image_pil = Image.fromarray(image)
    mask_image_pil = Image.fromarray(mask_image).convert("L")
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    new_image = pipe(prompt=prompt, image=image_pil, mask_image=mask_image_pil,
                     height=height, width=width, num_inference_steps=steps,
                     guidance_scale=guidance_scale, callback=None).images[0]

    return new_image


def read_image(image_path, max_size=512):
    create_crop_transforms(height=224, width=224)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # crop image
    height, width = image.shape[:2]
    height = height if height < max_size else max_size
    width = width if width < max_size else max_size
    transform = create_crop_transforms(height=height, width=width)
    image = transform(image=image)["image"]
    
    original_shape = image.shape
    new_height = find_nearest_multiple(original_shape[0], multiple=8)
    new_width = find_nearest_multiple(original_shape[1], multiple=8)
    new_image = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    new_image[:original_shape[0], :original_shape[1]] = image

    mask_image = np.zeros_like(image)

    del transform
    del image
    gc.collect()

    return new_image, mask_image, original_shape


def func(image_path, save_path, crop_save_path, step=50, max_size=1024, pipe=None):
    image, mask_image, original_shape = read_image(image_path, max_size)
    new_image = stable_diffusion_inpainting(pipe, image, mask_image, prompt='', steps=step,
                                          height=image.shape[0],
                                          width=image.shape[1],
                                          seed=2023, guidance_scale=7.5)
    new_image = new_image.crop(box=(0, 0, original_shape[1], original_shape[0]))
    new_image.save(save_path)
    if not os.path.exists(crop_save_path):
        image = Image.fromarray(image).crop(box=(0, 0, original_shape[1], original_shape[0]))
        image.save(crop_save_path)


def process_images(image_paths, image_root, save_root, crop_root, device_id, step=50, max_size=1024):
   
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/data/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(f'cuda:{device_id}')
    print(f"Process {device_id} initialized pipeline successfully")
    
    failed_num = 0
   
    pbar = tqdm(total=len(image_paths), desc=f'GPU-{device_id}', position=device_id)
    
    for image_path in image_paths:
        
        rel_path = os.path.relpath(image_path, image_root)
        save_path = os.path.join(save_root, os.path.splitext(rel_path)[0] + '.png')
        crop_save_path = os.path.join(crop_root, os.path.splitext(rel_path)[0] + '.png')
        
       
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)
        
        
        if os.path.exists(save_path) and os.path.exists(crop_save_path):
            pbar.update(1)
            continue
                
        try:
            func(image_path, save_path, crop_save_path, step=step, max_size=max_size, pipe=pipe)
            pbar.update(1)
        except Exception as e:
            failed_num += 1
            print(f'Failed to process {image_path}: {str(e)}')
            pbar.update(1)
    
    pbar.close()
    return failed_num



def find_images(dir_path):
    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


if __name__ == '__main__':
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    
    step = 50
    inpainting_dir = 'full_inpainting'
    
    
    base_input_root = '/data/RRDataset_final'
    base_output_root = '/data/RRDataset_test_DRCT_final'
    
    
    subfolders = ['original', 'transfer', 'redigital']
    categories = ['ai', 'real']
    
    
    all_image_paths = []
    for subfolder in subfolders:
        for category in categories:
            input_dir = os.path.join(base_input_root, subfolder, category)
            if os.path.exists(input_dir):
                image_paths = find_images(input_dir)
                all_image_paths.extend(image_paths)
                print(f'Found {len(image_paths)} images in {input_dir}')
    
    print(f'Total found {len(all_image_paths)} images')
    
    
    chunks = np.array_split(all_image_paths, num_gpus)
    print(f"Splitting {len(all_image_paths)} images into {num_gpus} chunks")
    
    
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=num_gpus)
    
    
    tasks = []
    for gpu_id in range(num_gpus):
        task = pool.apply_async(
            process_images,
            args=(chunks[gpu_id], base_input_root, base_output_root, base_output_root + '_crop', gpu_id, step, 1024)
        )
        tasks.append(task)
    
    
    total_failed = 0
    for task in tasks:
        failed_num = task.get()
        total_failed += failed_num
    
    pool.close()
    pool.join()
    
    print(f'All processes finished! Total processed: {len(all_image_paths)}, Total failed: {total_failed}')
import os
from PIL import Image

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_image_files(directory):
    """Get all image files from directory"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

def resize_image(image, target_size):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    ratio = min(target_size[0]/width, target_size[1]/height)
    new_size = (int(width*ratio), int(height*ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS) 
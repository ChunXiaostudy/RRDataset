from PIL import Image
import os
import json
from .utils import create_directory_if_not_exists, get_image_files, resize_image

class ImageToPdfConverter:
    def __init__(self):
        # A4 size (pixels, 300DPI)
        self.A4_SIZE = (2480, 3508)
        self.IMAGES_PER_PAGE = 12  # 12 images per page
        
        # Define precise image dimensions and margins
        self.IMAGE_WIDTH = 700  # Fixed image width
        self.IMAGE_HEIGHT = 700  # Fixed image height
        self.HORIZONTAL_MARGIN = 140  # Horizontal margin
        self.VERTICAL_MARGIN = 100    # Vertical margin for 4 rows
        self.HORIZONTAL_SPACING = 100 # Horizontal spacing between images
        self.VERTICAL_SPACING = 69    # Vertical spacing for 4 rows
        
        # Calculate fixed image positions (3 columns × 4 rows)
        self.IMAGE_POSITIONS = [
            # First row
            (self.HORIZONTAL_MARGIN, self.VERTICAL_MARGIN),
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING, self.VERTICAL_MARGIN),
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2, self.VERTICAL_MARGIN),
            
            # Second row
            (self.HORIZONTAL_MARGIN, self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING)),
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING, 
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING)),
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING)),
             
            # Third row
            (self.HORIZONTAL_MARGIN, self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2),
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING, 
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2),
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2),
             
            # Fourth row
            (self.HORIZONTAL_MARGIN, self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3),
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING, 
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3),
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3)
        ]
        
    def convert(self, input_dir, output_dir):
        """Convert input images to PDF with 12 images per page layout"""
        create_directory_if_not_exists(output_dir)
        
        # Get all image files
        image_files = get_image_files(input_dir)
        if not image_files:
            raise Exception("No image files found")

        # Create list for all pages
        pages = []
        current_page = Image.new('RGB', self.A4_SIZE, 'white')
        image_count = 0
        
        # Create image position mapping dictionary
        image_mapping = {}
        current_page_num = 1

        for img_file in image_files:
            # Open and resize image to fixed dimensions
            with Image.open(os.path.join(input_dir, img_file)) as img:
                # Resize image while maintaining aspect ratio
                img = resize_image(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
                
                # Get current image position
                position = self.IMAGE_POSITIONS[image_count % self.IMAGES_PER_PAGE]
                
                # Paste image at fixed position
                current_page.paste(img, position)
                
                # Save image mapping information
                image_mapping[f"{current_page_num}_{(image_count % self.IMAGES_PER_PAGE) + 1}"] = {
                    'original_name': os.path.splitext(img_file)[0],  # Without extension
                    'page': current_page_num,
                    'position': (image_count % self.IMAGES_PER_PAGE) + 1
                }
                
                image_count += 1

                # If page is full, add to pages list and create new page
                if image_count % self.IMAGES_PER_PAGE == 0:
                    pages.append(current_page)
                    current_page = Image.new('RGB', self.A4_SIZE, 'white')
                    current_page_num += 1

        # Add last page if it has any images
        if image_count % self.IMAGES_PER_PAGE != 0:
            pages.append(current_page)

        # Save all pages to single PDF file
        if pages:
            output_path = os.path.join(output_dir, 'combined_output.pdf')
            pages[0].save(output_path, save_all=True, append_images=pages[1:])
            
            # Save image mapping information to JSON file
            mapping_file = os.path.join(output_dir, 'image_mapping.json')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(image_mapping, f, ensure_ascii=False, indent=2)
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import json
from .utils import create_directory_if_not_exists
Image.MAX_IMAGE_PIXELS = None  # Disable image size limit

class PdfToImageConverter:
    def __init__(self):
        # A4 size (pixels, 300DPI)
        self.A4_SIZE = (2480, 3508)
        self.IMAGES_PER_PAGE = 12  # 12 images per page
        
        # Define precise image dimensions and margins (matching image_to_pdf.py)
        self.IMAGE_WIDTH = 700  # Fixed image width
        self.IMAGE_HEIGHT = 700  # Fixed image height
        self.HORIZONTAL_MARGIN = 140  # Horizontal margin
        self.VERTICAL_MARGIN = 100    # Vertical margin for 4 rows
        self.HORIZONTAL_SPACING = 100 # Horizontal spacing between images
        self.VERTICAL_SPACING = 69    # Vertical spacing for 4 rows
        
        # Define precise cropping regions (3 columns × 4 rows)
        self.CROP_REGIONS = [
            # First row
            (self.HORIZONTAL_MARGIN, 
             self.VERTICAL_MARGIN, 
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT),
            
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 2 + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT),
            
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 3 + self.HORIZONTAL_SPACING * 2,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT),
            
            # Second row
            (self.HORIZONTAL_MARGIN,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING),
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 2 + self.VERTICAL_SPACING),
            
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING),
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 2 + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 2 + self.VERTICAL_SPACING),
            
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING),
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 3 + self.HORIZONTAL_SPACING * 2,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 2 + self.VERTICAL_SPACING),
             
            # Third row
            (self.HORIZONTAL_MARGIN,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 3 + self.VERTICAL_SPACING * 2),
            
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 2 + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 3 + self.VERTICAL_SPACING * 2),
            
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 2,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 3 + self.HORIZONTAL_SPACING * 2,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 3 + self.VERTICAL_SPACING * 2),
             
            # Fourth row
            (self.HORIZONTAL_MARGIN,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 4 + self.VERTICAL_SPACING * 3),
            
            (self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 2 + self.HORIZONTAL_SPACING,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 4 + self.VERTICAL_SPACING * 3),
            
            (self.HORIZONTAL_MARGIN + (self.IMAGE_WIDTH + self.HORIZONTAL_SPACING) * 2,
             self.VERTICAL_MARGIN + (self.IMAGE_HEIGHT + self.VERTICAL_SPACING) * 3,
             self.HORIZONTAL_MARGIN + self.IMAGE_WIDTH * 3 + self.HORIZONTAL_SPACING * 2,
             self.VERTICAL_MARGIN + self.IMAGE_HEIGHT * 4 + self.VERTICAL_SPACING * 3)
        ]
        
    def convert(self, pdf_path, output_dir):
        """Extract images from PDF with original layout"""
        create_directory_if_not_exists(output_dir)
        
        print("Converting PDF to images, please wait...")
        
        try:
            # Try to load image mapping information
            pdf_dir = os.path.dirname(pdf_path)
            mapping_file = os.path.join(pdf_dir, 'image_mapping.json')
            image_mapping = {}
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    image_mapping = json.load(f)
            
            # Open PDF document
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            print(f"PDF has {total_pages} pages, starting extraction...")
            
            # Calculate scaling to match A4 size
            zoom_x = self.A4_SIZE[0] / pdf_document[0].rect.width
            zoom_y = self.A4_SIZE[1] / pdf_document[0].rect.height
            matrix = fitz.Matrix(zoom_x, zoom_y)
            
            for page_num, page in enumerate(pdf_document, 1):
                print(f"Processing page {page_num}/{total_pages}...")
                
                try:
                    pix = page.get_pixmap(
                        matrix=matrix,
                        alpha=False,
                        colorspace="rgb"
                    )
                    img_data = pix.tobytes("png")
                    
                    with Image.open(io.BytesIO(img_data)) as page_image:
                        if page_image.size != self.A4_SIZE:
                            page_image = page_image.resize(self.A4_SIZE, Image.Resampling.LANCZOS)
                            
                        for i, crop_region in enumerate(self.CROP_REGIONS, 1):
                            image = page_image.crop(crop_region)
                            
                            # Get original filename (if exists)
                            mapping_key = f"{page_num}_{i}"
                            if mapping_key in image_mapping:
                                original_name = image_mapping[mapping_key]['original_name']
                                output_filename = f"redigital_{original_name}.jpg"
                            else:
                                output_filename = f'{page_num}_{i}.jpg'
                            
                            # Save cropped image
                            output_path = os.path.join(output_dir, output_filename)
                            image.save(
                                output_path,
                                'JPEG',
                                quality=95,
                                optimize=True,
                                subsampling=0
                            )
                            print(f"Saved image: {output_filename}")
                            
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
                
            pdf_document.close()
                
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
            
        print("Image extraction completed!")
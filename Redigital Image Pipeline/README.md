# Image-PDF Converter

A Python-based tool for batch processing images and PDFs. This tool provides functionality for converting multiple images into a single PDF with a specific layout, and extracting individual images from such PDFs while preserving original image names.

## Features

- Convert multiple images into a single PDF
  - Automatically resize images to uniform dimensions
  - Arrange 12 images per A4 page (3 columns × 4 rows)
  - Maintain consistent spacing and margins
  - Preserve original image names for later extraction

- Extract images from PDF
  - Extract individual images from PDF pages
  - Maintain original image quality
  - Restore original image names with "redigital_" prefix
  - Support high-resolution output

## Project Structure

```
image-pdf-converter/
├── src/
│   ├── __init__.py
│   ├── image_to_pdf.py      # Image to PDF conversion module
│   ├── pdf_to_images.py     # PDF to images extraction module
│   └── utils.py             # Utility functions
├── input_images/            # Directory for input images
├── output/
│   ├── pdf/                 # Output directory for PDF files
│   └── images/              # Output directory for extracted images
├── requirements.txt         # Project dependencies
├── main.py                 # Main program entry point
└── README.md               # Project documentation
```

## Detailed Component Description

### Main Program (main.py)
- Entry point of the application
- Provides interactive command-line interface
- Manages input/output directory structures
- Handles error cases and user feedback

### Image to PDF Converter (image_to_pdf.py)
- Converts multiple images into a single PDF
- Features:
  - Maintains consistent A4 page layout (2480×3508 pixels at 300 DPI)
  - Resizes images to 700×700 pixels while preserving aspect ratio
  - Arranges 12 images per page in a 3×4 grid
  - Creates mapping file to preserve original image names
  - Supports common image formats (PNG, JPG, JPEG, BMP, GIF)

### PDF to Images Extractor (pdf_to_images.py)
- Extracts images from PDF files
- Features:
  - Precise image extraction using predefined crop regions
  - High-quality output (95% JPEG quality)
  - Restores original image names with "redigital_" prefix
  - Maintains image positioning and quality
  - Supports multi-page PDFs

### Utility Module (utils.py)
- Provides common utility functions:
  - Directory creation and management
  - Image file filtering
  - Image resizing with aspect ratio preservation

## Requirements

- Python 3.8+
- Dependencies:
  - Pillow==10.2.0 (Image processing)
  - PyPDF2==3.0.1 (PDF handling)
  - PyMuPDF==1.23.8 (PDF processing)

## Installation


1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Preparing Input Images
- Place your input images in the `input_images` directory
- Supported formats: PNG, JPG, JPEG, BMP, GIF
- Images will be automatically resized while maintaining aspect ratio

### 2. Running the Program
```bash
python main.py
```

### 3. Available Operations

#### Option 1: Convert Images to PDF
- Converts all images in `input_images` directory to a single PDF
- Output: `output/pdf/combined_output.pdf`
- Also creates: `output/pdf/image_mapping.json` (preserves original filenames)
- Images are arranged 12 per page in a 3×4 grid
- Each image is resized to 700×700 pixels

#### Option 2: Extract Images from PDF
- Extracts images from the most recently created PDF
- Output: Individual JPG files in `output/images`
- Naming convention:
  - If original name available: `redigital_original_name.jpg`
  - Otherwise: `page_number_position.jpg`
- Maintains high image quality (95% JPEG quality)

#### Option 3: Exit
- Safely exits the program

## Output Specifications

### PDF Output
- Format: A4 (2480×3508 pixels at 300 DPI)
- Layout: 12 images per page (3 columns × 4 rows)
- Margins and Spacing:
  - Horizontal margin: 140 pixels
  - Vertical margin: 100 pixels
  - Horizontal spacing: 100 pixels
  - Vertical spacing: 69 pixels

### Extracted Images
- Format: JPEG
- Quality: 99%
- Size: 700×700 pixels
- Optimization: Enabled
- Color subsampling: Disabled for better quality

## Error Handling
- Validates input image formats
- Creates necessary directories automatically
- Provides clear error messages for common issues
- Continues processing despite individual image failures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

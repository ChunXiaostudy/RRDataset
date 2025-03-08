import os
from src.image_to_pdf import ImageToPdfConverter
from src.pdf_to_images import PdfToImageConverter

# Define fixed input/output paths
INPUT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "input_images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
PDF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pdf")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")

def main():
    while True:
        print("\nSelect operation:")
        print("1. Convert images to PDF")
        print("2. Extract images from PDF")
        print("3. Exit")
        
        choice = input("Enter option (1-3): ")
        
        if choice == '1':
            converter = ImageToPdfConverter()
            try:
                converter.convert(INPUT_IMAGE_DIR, PDF_OUTPUT_DIR)
                print(f"Conversion completed! PDF file saved to: {PDF_OUTPUT_DIR}")
            except Exception as e:
                print(f"Conversion failed: {str(e)}")
                
        elif choice == '2':
            try:
                pdf_files = [f for f in os.listdir(PDF_OUTPUT_DIR) if f.endswith('.pdf')]
                if not pdf_files:
                    print("No PDF files found!")
                    continue
                    
                latest_pdf = max([os.path.join(PDF_OUTPUT_DIR, f) for f in pdf_files], 
                               key=os.path.getctime)
                
                print(f"Found PDF file: {os.path.basename(latest_pdf)}")
                print("Starting image extraction, this may take a while...")
                
                converter = PdfToImageConverter()
                converter.convert(latest_pdf, IMAGE_OUTPUT_DIR)
                
            except Exception as e:
                print(f"Extraction failed: {str(e)}")
                
        elif choice == '3':
            break
        else:
            print("Invalid option, please try again")

if __name__ == "__main__":
    # Ensure required directories exist
    for directory in [INPUT_IMAGE_DIR, OUTPUT_DIR, PDF_OUTPUT_DIR, IMAGE_OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    main()
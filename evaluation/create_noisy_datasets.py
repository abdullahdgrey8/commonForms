import os
import glob
import random
import numpy as np
import io  # <-- Import the io module
from PIL import Image, ImageFilter
import fitz  # This is PyMuPDF
import img2pdf

# --- Configuration ---
INPUT_DIR = "test_pdfs"
BLUR_OUTPUT_DIR = "blur_pdfs"
SALT_PEPPER_OUTPUT_DIR = "salt_pepper_pdfs"

# Noise Parameters
BLUR_RADIUS = 1.5  # Increase for more blur (e.g., 2.0, 3.0)
SALT_PEPPER_AMOUNT = 0.05  # Proportion of pixels to be affected by noise (0.05 = 5%)
DPI = 200  # Resolution for rendering PDF pages to images

def pil_image_to_pdf_buffer(image):
    """
    Converts a PIL Image to an in-memory buffer suitable for img2pdf.
    This is the key to fixing the error.
    """
    buffer = io.BytesIO()
    # Use PNG for lossless compression, ideal for text documents.
    # If the images were photos, JPEG could be used to save space.
    image.save(buffer, format='PNG')
    buffer.seek(0) # Go to the start of the buffer so img2pdf can read it
    return buffer

def add_salt_pepper_noise(image, amount=SALT_PEPPER_AMOUNT):
    """
    Adds Salt & Pepper noise to a PIL Image.
    Salt = white pixels, Pepper = black pixels.
    """
    img_arr = np.array(image)
    row, col, ch = img_arr.shape
    num_salt = np.ceil(amount * img_arr.size * 0.5)
    num_pepper = np.ceil(amount * img_arr.size * 0.5)
    
    # Add Salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (row, col)]
    img_arr[coords[0], coords[1], :] = 255

    # Add Pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (row, col)]
    img_arr[coords[0], coords[1], :] = 0
    
    noisy_image = Image.fromarray(img_arr)
    return noisy_image

def process_pdfs():
    """
    Main function to read PDFs, apply noise, and save them using PyMuPDF.
    """
    # --- 1. Setup Directories ---
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create it and add your PDF files.")
        return

    os.makedirs(BLUR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SALT_PEPPER_OUTPUT_DIR, exist_ok=True)

    # --- 2. Find all PDF files ---
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    # --- 3. Process each PDF ---
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n--- Processing {pdf_name} ---")

        try:
            doc = fitz.open(pdf_path)
            zoom = DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            page_images = []
            print(f"Rendering PDF pages at {DPI} DPI...")
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images.append(img)

            # --- Create Blurred PDF ---
            print(f"Applying Gaussian Blur (radius={BLUR_RADIUS})...")
            blurred_pages = [page.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)) for page in page_images]
            
            # Convert each PIL Image to an in-memory PDF-compatible buffer
            blurred_buffers = [pil_image_to_pdf_buffer(p) for p in blurred_pages]
            
            blur_output_path = os.path.join(BLUR_OUTPUT_DIR, pdf_name)
            with open(blur_output_path, "wb") as f:
                f.write(img2pdf.convert(blurred_buffers))
            print(f"Saved blurred PDF to {blur_output_path}")

            # --- Create Salt & Pepper PDF ---
            print(f"Applying Salt & Pepper Noise (amount={SALT_PEPPER_AMOUNT})...")
            noisy_pages = [add_salt_pepper_noise(page, amount=SALT_PEPPER_AMOUNT) for page in page_images]

            # Convert each PIL Image to an in-memory PDF-compatible buffer
            noisy_buffers = [pil_image_to_pdf_buffer(p) for p in noisy_pages]

            sp_output_path = os.path.join(SALT_PEPPER_OUTPUT_DIR, pdf_name)
            with open(sp_output_path, "wb") as f:
                f.write(img2pdf.convert(noisy_buffers))
            print(f"Saved salt & pepper PDF to {sp_output_path}")
            
            doc.close()

        except Exception as e:
            print(f"An error occurred while processing {pdf_name}: {e}")

    print("\n--- All PDFs processed! ---")


if __name__ == "__main__":
    process_pdfs()
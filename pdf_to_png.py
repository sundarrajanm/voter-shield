import os
from pdf2image import convert_from_path
from PIL import Image
import time

START_PAGE = 3
END_PAGE = 43

def convert_pdfs_to_png(pdf_dir: str, png_dir: str, dpi: int):
    """
    Reads PDFs from pdf_dir and writes page-level PNGs to png_dir
    """
    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    for file in sorted(os.listdir(pdf_dir)):
        if file.lower().endswith(".pdf"):
            input_pdf_path = os.path.join(pdf_dir, file)
            print(f"🚀 Converting {input_pdf_path} to PNG images...  ")
            pages = convert_from_path(input_pdf_path, dpi=dpi)

            for i, page in enumerate(pages):
                if i >= START_PAGE - 1 and i < END_PAGE:
                    print(f"🤖 Writing page {i+1:02d} as image.")
                    output_png_path = os.path.join(png_dir, f"{file.replace('.pdf', '')}_page_{i+1:02d}.png")
                    page.save(output_png_path, "PNG")

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Total time taken: {elapsed_time:.3f} seconds.")
    print("✅ PDF to PNG conversion completed.")

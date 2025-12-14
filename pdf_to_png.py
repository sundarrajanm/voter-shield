import os
from pdf2image import convert_from_path
from PIL import Image
import time

from logger import setup_logger
logger = setup_logger()

START_PAGE = 3
END_PAGE = 43

def convert_pdfs_to_png(pdf_dir: str, png_dir: str, dpi: int, progress=None):
    """
    Reads PDFs from pdf_dir and writes page-level PNGs to png_dir
    """

    # Record the start time
    start_time = time.perf_counter() # Or time.time() for less precision

    for file in sorted(os.listdir(pdf_dir)):
        if file.lower().endswith(".pdf"):
            input_pdf_path = os.path.join(pdf_dir, file)

            logger.info(f"🚀 Converting {input_pdf_path} to PNG images...  ")
            pages = convert_from_path(input_pdf_path, dpi=dpi)
            pages = enumerate(pages)

            task = None
            if progress:
                task = progress.add_task("1️⃣ Converting PDF to PNGs", total=END_PAGE - START_PAGE + 1)

            for i, page in pages:
                if i >= START_PAGE - 1 and i < END_PAGE:
                    output_png_path = os.path.join(png_dir, f"{file.replace('.pdf', '')}_page_{i+1:02d}.png")
                    page.save(output_png_path, "PNG")
                    if progress:
                        progress.advance(task)                        

    # Record the end time
    end_time = time.perf_counter() # Or time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    logger.info(f"Total time taken: {elapsed_time:.3f} seconds.")
    logger.info("✅ PDF to PNG conversion completed.")
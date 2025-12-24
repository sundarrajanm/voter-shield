import os
from pdf2image import convert_from_path
from PIL import Image
import time

from logger import setup_logger
logger = setup_logger()

# def convert_pdfs_to_png(pdf_dir: str, png_dir: str, dpi: int, progress=None):
#     """
#     Reads PDFs from pdf_dir and writes page-level PNGs to png_dir
#     """

#     # Record the start time
#     start_time = time.perf_counter() # Or time.time() for less precision

#     task1 = None
#     files = sorted(os.listdir(pdf_dir))
#     if progress:
#         task1 = progress.add_task("1️⃣ Working on multiple PDFs", total=len(files) - 1) # -1 to .DS_Store (MacOS)

#     for file in files:
#         if file.lower().endswith(".pdf"):
#             if progress:
#                 progress.advance(task1)

#             input_pdf_path = os.path.join(pdf_dir, file)

#             logger.info(f"🚀 Converting {input_pdf_path} to PNG images...  ")
#             pages = convert_from_path(input_pdf_path, dpi=dpi)
#             pages = enumerate(pages)

#             task2 = None
#             if progress:
#                 task2 = progress.add_task("⌛️ Converting single PDF to PNGs", total=END_PAGE - START_PAGE + 1)

#             for i, page in pages:
#                 if i >= START_PAGE - 1 and i < END_PAGE:
#                     output_png_path = os.path.join(png_dir, f"{file.replace('.pdf', '')}_page_{i+1:02d}.png")
#                     page.save(output_png_path, "PNG")
#                     if progress:
#                         progress.advance(task2)                        

#     # Record the end time
#     end_time = time.perf_counter() # Or time.time()

#     # Calculate the elapsed time
#     elapsed_time = end_time - start_time

#     # Print the elapsed time
#     logger.info(f"Total time taken: {elapsed_time:.3f} seconds.")
#     logger.info("✅ PDF to PNG conversion completed.")

from pdf2image import convert_from_path

START_PAGE = 3 # skip first 2 pages

def _convert_single_pdf(
    pdf_path: str,
    jpg_dir: str,
    dpi: int,
    progress=None
):
    logger.info(f"🚀 Converting {pdf_path}")
    pages = convert_from_path(pdf_path,
                              dpi=dpi,
                              fmt="jpeg",
                              jpegopt={"quality": 95},
                              first_page=START_PAGE)

    total_pages = len(pages) - 1 # skip last page

    task_child = None
    if progress:
        task_child = progress.add_task(
            f"⌛️ {os.path.basename(pdf_path)}",
            total=total_pages
        )

    for i, page in enumerate(pages):
        if i < total_pages:
            out = os.path.join(
                jpg_dir,
                f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{i+1:02d}.jpg"
            )
            page.save(out, "JPEG")
            if progress and task_child:
                progress.advance(task_child)

from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_pdfs_to_jpg(
    pdf_dir: str,
    jpg_dir: str,
    dpi: int,
    progress=None,
    max_workers=4,
    limit=None
):
    start_time = time.perf_counter()

    os.makedirs(jpg_dir, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )

    if limit is not None:
        pdf_files = pdf_files[:limit]
        
    if not pdf_files:
        logger.warning("No PDFs found")
        return

    task_parent = None

    if progress:
        task_parent = progress.add_task(
            "1️⃣ Converting PDFs to JPGs",
            total=len(pdf_files)
        )
    
    logger.info(f"🔄 Starting conversion of {len(pdf_files)} PDFs using {max_workers} threads")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(
                _convert_single_pdf,
                os.path.join(pdf_dir, file),
                jpg_dir,
                dpi,
                progress
            ): file
            for file in pdf_files
        }

        completed = 0
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            future.result()
            completed += 1
            if progress and task_parent:
                progress.advance(task_parent)

    if progress and task_parent:
        progress.update(task_parent, completed=len(pdf_files))

    elapsed = time.perf_counter() - start_time
    logger.info(f"⏱️ PDF → JPG completed in {elapsed:.2f} seconds")

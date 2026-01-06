# Docker Setup for Electoral Roll Processing

This project can be run inside a Docker container to ensure all dependencies (OCRs, system libraries) are correctly installed.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (usually comes with Docker Desktop).

## Setup

1. **Build the Docker image:**

   ```bash
   docker-compose build
   ```

   This may take a few minutes as it installs Python dependencies and system packages (Tesseract OCR, etc.).

## Usage

### 1. Process All PDFs
To process all PDFs located in the `pdfs/` directory:

```bash
docker-compose run --rm app python main.py
```

### 2. Process Specific PDFs
To process specific PDF files inside the `pdfs/` directory:

```bash
docker-compose run --rm app python main.py pdfs/your-file.pdf
```

### 3. Run Specific Steps
You can run specific steps like just extraction or just OCR:

```bash
# Only extract images
docker-compose run --rm app python main.py --step extract

# Only run OCR (on already extracted folders)
docker-compose run --rm app python main.py --step ocr
```

### 4. Interactive Shell
If you want to explore the container or run commands manually:

```bash
docker-compose run --rm --entrypoint bash app
```

## Configuration

- The application uses values from the `.env` file.
- The `pdfs/`, `extracted/`, and `logs/` directories are mounted from your host machine. Any files placed in `pdfs/` will be visible to the container, and results in `extracted/` will persist on your machine.

## Troubleshooting

- **Permissions:** If you encounter permission errors with the generated files, ensure your user has write access to the `extracted/` directory.
- **Rebuild:** If you change `requirements.txt`, remember to rebuild:
  ```bash
  docker-compose build
  ```

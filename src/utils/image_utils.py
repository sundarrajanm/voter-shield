"""
Image processing utility functions.

Common operations for image manipulation in the electoral roll processing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Load image from file with proper Unicode path handling.
    
    Args:
        path: Path to image file
        flags: OpenCV imread flags
    
    Returns:
        Loaded image as numpy array, or None if failed
    """
    path = Path(path)
    if not path.exists():
        return None
    
    # Use cv2.imdecode for Unicode path support
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        # Fallback to regular imread
        return cv2.imread(str(path), flags)


def save_image(
    image: np.ndarray,
    path: Path,
    quality: int = 100,
    compression: int = 0
) -> bool:
    """
    Save image to file with proper Unicode path handling.
    
    Args:
        image: Image to save
        path: Output path
        quality: JPEG quality (0-100)
        compression: PNG compression level (0-9)
    
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = path.suffix.lower()
    
    try:
        if ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            params = []
        
        # Use cv2.imencode for Unicode path support
        success, data = cv2.imencode(ext, image, params)
        if success:
            data.tofile(str(path))
            return True
        return False
    except Exception:
        # Fallback to regular imwrite
        return cv2.imwrite(str(path), image)


def crop_relative(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> np.ndarray:
    """
    Crop image using relative coordinates (0-1).
    
    Args:
        image: Source image
        x1, y1: Top-left corner (relative)
        x2, y2: Bottom-right corner (relative)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Convert to pixel coordinates
    X1 = int(round(x1 * w))
    Y1 = int(round(y1 * h))
    X2 = int(round(x2 * w))
    Y2 = int(round(y2 * h))
    
    # Clamp to valid range
    X1 = max(0, min(w - 1, X1))
    X2 = max(1, min(w, X2))
    Y1 = max(0, min(h - 1, Y1))
    Y2 = max(1, min(h, Y2))
    
    return image[Y1:Y2, X1:X2]


def crop_roi(
    image: np.ndarray,
    roi: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Crop image using ROI tuple.
    
    Args:
        image: Source image
        roi: (x1, y1, x2, y2) relative coordinates
    
    Returns:
        Cropped image
    """
    return crop_relative(image, *roi)


def estimate_skew(gray: np.ndarray) -> float:
    """
    Estimate document skew angle using Hough lines.
    
    Args:
        gray: Grayscale image
    
    Returns:
        Estimated skew angle in degrees
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(50, gray.shape[1] // 3),
        maxLineGap=10
    )
    
    if lines is None:
        return 0.0
    
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep only near-horizontal lines
        if -30 <= angle <= 30:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Use median for robustness
    return float(np.median(angles))


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image to correct skew.
    
    Args:
        image: Source image (color or grayscale)
        angle: Skew angle in degrees
    
    Returns:
        Deskewed image
    """
    if abs(angle) < 0.2:
        return image
    
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_for_ocr(
    image: np.ndarray,
    scale: float = 2.0,
    denoise: bool = True,
    deskew_image: bool = True
) -> np.ndarray:
    """
    Preprocess image for OCR.
    
    Standard preprocessing pipeline:
    1. Convert to grayscale
    2. Deskew
    3. Upscale
    4. Denoise
    5. Normalize contrast
    
    Args:
        image: Source image (color or grayscale)
        scale: Scale factor for upscaling
        denoise: Whether to apply denoising
        deskew_image: Whether to apply deskew correction
    
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Deskew
    if deskew_image:
        angle = estimate_skew(gray)
        if abs(angle) > 0.2:
            gray = deskew(gray, angle)
    
    # Upscale
    if scale != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    
    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=8,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    # Normalize contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    return gray


def preprocess_for_epic(image: np.ndarray) -> np.ndarray:
    """
    Specialized preprocessing for EPIC number extraction.
    
    Args:
        image: Color image of EPIC region
    
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=8, templateWindowSize=7, searchWindowSize=21
    )
    return gray


def preprocess_for_serial(image: np.ndarray) -> np.ndarray:
    """
    Specialized preprocessing for serial number extraction.
    
    Args:
        image: Color image of serial number region
    
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )
    # Invert if text is light on dark background
    if float(np.mean(gray)) > 180:
        gray = 255 - gray
    return gray


def extract_grid_lines(binary: np.ndarray, hscale: int = 25, vscale: int = 25) -> np.ndarray:
    """
    Extract horizontal and vertical grid lines from binary image.
    
    Args:
        binary: Binary (thresholded) image
        hscale: Horizontal line kernel scale factor
        vscale: Vertical line kernel scale factor
    
    Returns:
        Binary image with grid lines
    """
    h, w = binary.shape[:2]
    
    # Horizontal lines
    h_kernel_len = max(10, w // hscale)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horiz = cv2.erode(binary, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, h_kernel, iterations=1)
    
    # Vertical lines
    v_kernel_len = max(10, h // vscale)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vert = cv2.erode(binary, v_kernel, iterations=1)
    vert = cv2.dilate(vert, v_kernel, iterations=1)
    
    # Combine
    grid = cv2.bitwise_or(horiz, vert)
    
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return grid


def resize_to_canonical(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize image to canonical dimensions.
    
    Args:
        image: Source image
        width: Target width
        height: Target height
    
    Returns:
        Resized image
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_roi_overlay(
    image: np.ndarray,
    rois: dict[str, Tuple[float, float, float, float]],
    colors: Optional[dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw ROI overlays on image for debugging.
    
    Args:
        image: Source image
        rois: Dictionary mapping ROI name to (x1, y1, x2, y2) coordinates
        colors: Optional dictionary mapping ROI name to BGR color
    
    Returns:
        Image with ROI rectangles drawn
    """
    output = image.copy()
    h, w = output.shape[:2]
    
    default_colors = {
        "epic": (0, 255, 0),      # Green
        "serial": (255, 0, 0),    # Blue
        "house": (0, 0, 255),     # Red
        "name": (255, 255, 0),    # Cyan
    }
    colors = colors or default_colors
    
    for name, roi in rois.items():
        x1, y1, x2, y2 = roi
        X1, Y1 = int(x1 * w), int(y1 * h)
        X2, Y2 = int(x2 * w), int(y2 * h)
        
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(output, (X1, Y1), (X2, Y2), color, 2)
        cv2.putText(
            output, name, (X1, Y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
    return output

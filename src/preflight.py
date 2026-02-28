"""
Project Akshara: Pass 0 - Pre-flight Processing
===============================================

Deterministic image preprocessing to optimize PDFs for LLM extraction.
No LLM calls - pure image processing.

Operations:
- PDF to image conversion
- DPI normalization
- Deskewing
- Contrast enhancement
- Noise reduction
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import tempfile

# Type hints only - not evaluated at runtime
if TYPE_CHECKING:
    import fitz
    import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from PIL import Image


@dataclass
class PageImage:
    """Processed page image."""
    page_number: int
    image_path: Path
    original_dpi: int
    processed_dpi: int
    width: int
    height: int
    quality: str  # "good", "degraded", "poor"
    enhancements_applied: List[str]


@dataclass
class PreflightResult:
    """Result of pre-flight processing."""
    pdf_path: Path
    total_pages: int
    output_dir: Path
    pages: List[PageImage]
    warnings: List[str]
    errors: List[str]


class PreflightProcessor:
    """
    Pre-flight image processor for PDF preparation.

    Usage:
        processor = PreflightProcessor(target_dpi=300)
        result = processor.process_pdf("/path/to/book.pdf", "/output/dir")

        for page in result.pages:
            print(f"Page {page.page_number}: {page.image_path}")
    """

    def __init__(
        self,
        target_dpi: int = 300,
        enable_deskew: bool = True,
        enable_contrast: bool = True,
        enable_denoise: bool = True,
        image_format: str = "png",
        verbose: bool = True
    ):
        """
        Initialize pre-flight processor.

        Args:
            target_dpi: Target DPI for output images (minimum 300 recommended)
            enable_deskew: Apply deskewing to straighten pages
            enable_contrast: Apply contrast enhancement
            enable_denoise: Apply noise reduction
            image_format: Output image format ("png" or "jpg")
            verbose: Print progress messages
        """
        self.target_dpi = max(target_dpi, 300)  # Minimum 300 DPI
        self.enable_deskew = enable_deskew and OPENCV_AVAILABLE
        self.enable_contrast = enable_contrast and OPENCV_AVAILABLE
        self.enable_denoise = enable_denoise and OPENCV_AVAILABLE
        self.image_format = image_format
        self.verbose = verbose

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

    def _log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[Preflight] {message}")

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        page_range: Optional[Tuple[int, int]] = None,
        output_name: Optional[str] = None,
    ) -> PreflightResult:
        """
        Process a PDF file.

        Args:
            pdf_path: Path to input PDF
            output_dir: Directory for output images
            page_range: Optional (start, end) page range (1-indexed, inclusive)

        Returns:
            PreflightResult with processed page information
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = output_name or pdf_path.stem

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._log(f"Processing: {pdf_path.name}")

        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        self._log(f"Total pages: {total_pages}")

        # Determine page range
        start_page = 0
        end_page = total_pages
        if page_range:
            start_page = max(0, page_range[0] - 1)
            end_page = min(total_pages, page_range[1])

        pages: List[PageImage] = []
        warnings: List[str] = []
        errors: List[str] = []

        # Process each page
        for page_num in range(start_page, end_page):
            try:
                # Check if image already exists
                expected_output = output_dir / f"{output_name}_page_{page_num + 1:04d}.{self.image_format}"
                if expected_output.exists():
                    # Quick skip
                    pages.append(PageImage(
                        page_number=page_num + 1,
                        image_path=expected_output,
                        original_dpi=72,
                        processed_dpi=self.target_dpi,
                        width=0, # Unknown without loading, but acceptable for skip
                        height=0,
                        quality="good", # Assume good if skipping
                        enhancements_applied=["skipped (cached)"]
                    ))
                    continue

                page_result = self._process_page(
                    doc, page_num, output_dir, output_name
                )
                pages.append(page_result)

                if page_result.quality == "poor":
                    warnings.append(f"Page {page_num + 1}: Poor quality detected")

            except Exception as e:
                errors.append(f"Page {page_num + 1}: {str(e)}")
                self._log(f"Error on page {page_num + 1}: {e}")

        doc.close()

        self._log(f"Processed {len(pages)} pages, {len(warnings)} warnings, {len(errors)} errors")

        return PreflightResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            output_dir=output_dir,
            pages=pages,
            warnings=warnings,
            errors=errors
        )

    def _process_page(
        self,
        doc: "fitz.Document",
        page_num: int,
        output_dir: Path,
        book_name: str
    ) -> PageImage:
        """Process a single page."""
        page = doc[page_num]

        # Calculate zoom for target DPI
        # PDF default is 72 DPI
        zoom = self.target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Render page to image
        pix = page.get_pixmap(matrix=mat)

        # Convert to numpy array if OpenCV available
        if OPENCV_AVAILABLE:
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img_data = img_data.reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            enhancements = []
            quality = self._assess_quality(img)

            # Apply enhancements based on quality
            if self.enable_contrast:
                img = self._enhance_contrast(img)
                enhancements.append("contrast")

            if self.enable_denoise and quality in ["degraded", "poor"]:
                img = self._denoise(img)
                enhancements.append("denoise")

            if self.enable_deskew:
                img, skew_angle = self._deskew(img)
                if abs(skew_angle) > 0.5:
                    enhancements.append(f"deskew({skew_angle:.1f}°)")

            # Save processed image
            output_path = output_dir / f"{book_name}_page_{page_num + 1:04d}.{self.image_format}"
            cv2.imwrite(str(output_path), img)

        else:
            # No OpenCV - save directly from PyMuPDF
            output_path = output_dir / f"{book_name}_page_{page_num + 1:04d}.{self.image_format}"
            pix.save(str(output_path))
            enhancements = []
            quality = "unknown"

        return PageImage(
            page_number=page_num + 1,
            image_path=output_path,
            original_dpi=72,
            processed_dpi=self.target_dpi,
            width=pix.width,
            height=pix.height,
            quality=quality,
            enhancements_applied=enhancements
        )

    def _assess_quality(self, img: "np.ndarray") -> str:
        """Assess image quality."""
        if not OPENCV_AVAILABLE:
            return "unknown"

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        # 1. Contrast (standard deviation)
        contrast = np.std(gray)

        # 2. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 3. Brightness
        brightness = np.mean(gray)

        # Assess quality
        if contrast < 30 or laplacian_var < 100 or brightness < 50 or brightness > 230:
            return "poor"
        elif contrast < 50 or laplacian_var < 500:
            return "degraded"
        else:
            return "good"

    def _enhance_contrast(self, img: "np.ndarray") -> "np.ndarray":
        """Apply adaptive contrast enhancement."""
        if not OPENCV_AVAILABLE:
            return img

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _denoise(self, img: "np.ndarray") -> "np.ndarray":
        """Apply noise reduction."""
        if not OPENCV_AVAILABLE:
            return img

        # Use Non-local Means Denoising
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def _deskew(self, img: "np.ndarray") -> Tuple["np.ndarray", float]:
        """Deskew image and return angle."""
        if not OPENCV_AVAILABLE:
            return img, 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 100:
            return img, 0.0

        # Get minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if angle is significant but not too extreme
        if abs(angle) > 0.5 and abs(angle) < 15:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return img, angle

        return img, 0.0


def process_pdf(
    pdf_path: str,
    output_dir: str,
    **kwargs
) -> PreflightResult:
    """Convenience function to process a PDF."""
    processor = PreflightProcessor(**kwargs)
    return processor.process_pdf(pdf_path, output_dir)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Pre-flight processor ready.")
    print(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
    print(f"OpenCV available: {OPENCV_AVAILABLE}")

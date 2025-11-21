# app/utils/dpi_analyzer.py
from pathlib import Path
import PyPDF2
from PIL import Image
import fitz  # PyMuPDF
from typing import Optional
from app.core.logging import get_logger

logger = get_logger(__name__)


class DPIAnalyzer:
    """Analyzes PDF DPI and image quality"""
    
    @staticmethod
    def get_pdf_dpi(pdf_path: str) -> float:
        """
        Extract DPI from PDF by analyzing images in first page
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Average DPI of images in the PDF
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                logger.warning(f"PDF has no pages: {pdf_path}")
                return 0.0
            
            page = doc[0]
            images = page.get_images(full=True)
            
            if not images:
                # No images, try to estimate from page size
                # Assuming standard letter size (8.5 x 11 inches)
                rect = page.rect
                width_inches = rect.width / 72  # Convert points to inches
                height_inches = rect.height / 72
                
                # Use a default resolution estimate
                dpi = 72.0  # Default PDF resolution
                logger.info(f"No images found, using default DPI: {dpi}")
                return dpi
            
            dpi_values = []
            
            for img_index, img in enumerate(images[:5]):  # Check first 5 images
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image dimensions
                    img_width = base_image["width"]
                    img_height = base_image["height"]
                    
                    # Get display size on page
                    image_rects = page.get_image_rects(xref)
                    if image_rects:
                        rect = image_rects[0]
                        display_width = rect.width / 72  # Convert to inches
                        display_height = rect.height / 72
                        
                        if display_width > 0 and display_height > 0:
                            dpi_x = img_width / display_width
                            dpi_y = img_height / display_height
                            avg_dpi = (dpi_x + dpi_y) / 2
                            dpi_values.append(avg_dpi)
                
                except Exception as e:
                    logger.debug(f"Could not process image {img_index}: {e}")
                    continue
            
            doc.close()
            
            if dpi_values:
                avg_dpi = sum(dpi_values) / len(dpi_values)
                logger.info(f"Calculated average DPI: {avg_dpi:.2f}")
                return round(avg_dpi, 2)
            else:
                logger.warning("Could not calculate DPI from images, using default")
                return 72.0
                
        except Exception as e:
            logger.error(f"Error analyzing PDF DPI: {e}")
            return 0.0
    
    @staticmethod
    def estimate_pdf_quality(pdf_path: str) -> dict:
        """
        Estimate overall PDF quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        try:
            doc = fitz.open(pdf_path)
            
            total_images = 0
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                images = page.get_images(full=True)
                total_images += len(images)
            
            doc.close()
            
            return {
                "total_pages": total_pages,
                "total_images": total_images,
                "images_per_page": round(total_images / total_pages, 2) if total_pages > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error estimating PDF quality: {e}")
            return {
                "total_pages": 0,
                "total_images": 0,
                "images_per_page": 0
            }


# Global analyzer instance
dpi_analyzer = DPIAnalyzer()
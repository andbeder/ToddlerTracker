"""
Image Conversion Utility
Handles conversion of various image formats (especially HEIC) to JPEG for compatibility.
"""

import io
import os
import logging
from typing import Tuple, Optional
from PIL import Image
import pillow_heif

logger = logging.getLogger(__name__)

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

class ImageConverter:
    """Utility class for converting image formats."""

    def __init__(self):
        """Initialize image converter."""
        self.supported_formats = {
            'HEIC', 'HEIF', 'PNG', 'JPEG', 'JPG', 'BMP', 'TIFF', 'WEBP'
        }
        logger.info("Image converter initialized with HEIC support")

    def is_supported_format(self, filename: str) -> bool:
        """Check if the file format is supported."""
        if not filename:
            return False

        extension = os.path.splitext(filename)[1].upper().lstrip('.')
        return extension in self.supported_formats

    def detect_format(self, image_data: bytes) -> Optional[str]:
        """Detect image format from raw bytes."""
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                return img.format
        except Exception as e:
            logger.error(f"Error detecting image format: {e}")
            return None

    def convert_to_jpeg(self, image_data: bytes, quality: int = 95,
                       max_size: Tuple[int, int] = None) -> Tuple[bytes, bool]:
        """
        Convert image data to JPEG format.

        Args:
            image_data: Raw image bytes
            quality: JPEG quality (1-100)
            max_size: Optional tuple (width, height) to resize image

        Returns:
            Tuple of (converted_jpeg_bytes, was_converted)
        """
        try:
            # Open the image
            with Image.open(io.BytesIO(image_data)) as img:
                original_format = img.format
                logger.info(f"Processing {original_format} image")

                # Convert to RGB if necessary (HEIC might be in different color space)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if max_size is specified
                if max_size:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")

                # Convert to JPEG
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                jpeg_data = output_buffer.getvalue()

                was_converted = original_format != 'JPEG'
                if was_converted:
                    logger.info(f"Converted {original_format} to JPEG ({len(image_data)} -> {len(jpeg_data)} bytes)")
                else:
                    logger.info(f"Image already JPEG format")

                return jpeg_data, was_converted

        except Exception as e:
            logger.error(f"Error converting image to JPEG: {e}")
            # Return original data if conversion fails
            return image_data, False

    def process_upload(self, image_data: bytes, filename: str = None,
                      max_size: Tuple[int, int] = (2048, 2048)) -> Tuple[bytes, str, bool]:
        """
        Process uploaded image data for CompreFace compatibility.

        Args:
            image_data: Raw image bytes
            filename: Original filename (optional)
            max_size: Maximum dimensions to resize to

        Returns:
            Tuple of (processed_image_data, suggested_filename, was_converted)
        """
        try:
            # Detect format
            detected_format = self.detect_format(image_data)
            logger.info(f"Detected format: {detected_format} for file: {filename}")

            # Convert to JPEG
            jpeg_data, was_converted = self.convert_to_jpeg(image_data, max_size=max_size)

            # Generate appropriate filename
            if filename:
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}.jpg"
            else:
                new_filename = "converted_image.jpg"

            return jpeg_data, new_filename, was_converted

        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return image_data, filename or "image.jpg", False

    def validate_image(self, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate that image data is readable and suitable for face recognition.

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Check minimum size
                if img.size[0] < 50 or img.size[1] < 50:
                    return False, "Image too small (minimum 50x50 pixels)"

                # Check maximum size
                if img.size[0] > 10000 or img.size[1] > 10000:
                    return False, "Image too large (maximum 10000x10000 pixels)"

                # Check file size (rough estimate)
                if len(image_data) > 20 * 1024 * 1024:  # 20MB
                    return False, "Image file too large (maximum 20MB)"

                return True, "Image is valid"

        except Exception as e:
            return False, f"Invalid image: {str(e)}"


# Global instance
_image_converter = None

def get_image_converter() -> ImageConverter:
    """Get global image converter instance."""
    global _image_converter
    if _image_converter is None:
        _image_converter = ImageConverter()
    return _image_converter

def convert_heic_to_jpeg(image_data: bytes) -> Tuple[bytes, bool]:
    """Convenience function to convert HEIC to JPEG."""
    converter = get_image_converter()
    return converter.convert_to_jpeg(image_data)

def process_image_upload(image_data: bytes, filename: str = None) -> Tuple[bytes, str, bool]:
    """Convenience function to process image uploads."""
    converter = get_image_converter()
    return converter.process_upload(image_data, filename)
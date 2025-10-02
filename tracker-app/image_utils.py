"""
Image processing utilities for the Toddler Tracker application.
Handles thumbnail creation and image manipulation.
"""

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Optional, Tuple, Dict


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def create_thumbnail(image_data: bytes, size: Tuple[int, int] = (150, 150),
                        quality: int = 85) -> Optional[bytes]:
        """
        Create a thumbnail from image data.

        Args:
            image_data: Raw image bytes
            size: Target thumbnail size as (width, height)
            quality: JPEG quality (1-100)

        Returns:
            Thumbnail image bytes or None if error
        """
        try:
            # Open image from bytes
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Create thumbnail while maintaining aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)

            # Save as JPEG to bytes
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            output.seek(0)

            return output.getvalue()
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None

    @staticmethod
    def get_image_info(image_data: bytes) -> Optional[dict]:
        """
        Get information about an image.

        Args:
            image_data: Raw image bytes

        Returns:
            Dictionary with image info or None if error
        """
        try:
            image = Image.open(BytesIO(image_data))
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
        except Exception as e:
            print(f"Error getting image info: {e}")
            return None

    @staticmethod
    def resize_image(image_data: bytes, target_size: Tuple[int, int],
                    maintain_aspect: bool = True, quality: int = 85) -> Optional[bytes]:
        """
        Resize an image to target dimensions.

        Args:
            image_data: Raw image bytes
            target_size: Target size as (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            quality: JPEG quality (1-100)

        Returns:
            Resized image bytes or None if error
        """
        try:
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if maintain_aspect:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Save as JPEG to bytes
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            output.seek(0)

            return output.getvalue()
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None

    @staticmethod
    def create_detection_thumbnail(image_data: bytes, box: Dict,
                                   label: str = "", size: Tuple[int, int] = (300, 300),
                                   quality: int = 85) -> Optional[bytes]:
        """
        Create a thumbnail from image data with detection bounding box drawn.

        Args:
            image_data: Raw image bytes
            box: Bounding box dict with keys: x_min, y_min, x_max, y_max
            label: Text label to display above box
            size: Target thumbnail size as (width, height)
            quality: JPEG quality (1-100)

        Returns:
            Thumbnail image bytes with bounding box or None if error
        """
        try:
            # Open image from bytes
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Draw bounding box if provided
            if box and all(k in box for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                draw = ImageDraw.Draw(image)

                # Extract box coordinates
                x_min = int(box['x_min'])
                y_min = int(box['y_min'])
                x_max = int(box['x_max'])
                y_max = int(box['y_max'])

                # Draw rectangle (green with 3px width)
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline='#00FF00',
                    width=3
                )

                # Draw label if provided
                if label:
                    # Try to use a font, fall back to default if not available
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    except:
                        font = ImageFont.load_default()

                    # Get text bounding box for background
                    bbox = draw.textbbox((x_min, y_min - 25), label, font=font)

                    # Draw background rectangle for text
                    draw.rectangle(bbox, fill='#00FF00')

                    # Draw text
                    draw.text((x_min, y_min - 25), label, fill='#000000', font=font)

            # Create thumbnail while maintaining aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)

            # Save as JPEG to bytes
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            output.seek(0)

            return output.getvalue()
        except Exception as e:
            print(f"Error creating detection thumbnail: {e}")
            # Fall back to regular thumbnail without box
            return ImageProcessor.create_thumbnail(image_data, size, quality)

    @staticmethod
    def create_placeholder_svg(width: int = 100, height: int = 100,
                              text: str = "No Image", bg_color: str = "#e9ecef",
                              text_color: str = "#6c757d") -> str:
        """
        Create an SVG placeholder image.

        Args:
            width: SVG width
            height: SVG height
            text: Text to display
            bg_color: Background color
            text_color: Text color

        Returns:
            SVG string
        """
        return f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="{bg_color}"/>
            <text x="{width//2}" y="{height//2}" text-anchor="middle"
                  font-family="Arial" font-size="{min(width, height)//8}" fill="{text_color}">
                {text}
            </text>
        </svg>
        '''
"""
Image utilities for the Drawing Intelligence System.

Provides functions for image encoding, manipulation, and visualization.
"""

import base64
import io
from typing import List, Optional, Tuple
import numpy as np
import cv2
import hashlib
from PIL import Image


def encode_image_base64(image: np.ndarray, format: str = "PNG") -> str:
    """
    Encode numpy image to base64 string.

    Args:
        image: Numpy array (BGR or grayscale)
        format: Image format ('PNG', 'JPEG')

    Returns:
        Base64-encoded string

    Raises:
        ValueError: If image is invalid or encoding fails
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image: empty or None")

    # Convert to PIL Image
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image)
    elif len(image.shape) == 3:
        # Convert BGR to RGB for PIL
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        elif image.shape[2] == 4:
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            pil_image = Image.fromarray(rgba_image)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")

    # Encode to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    image_bytes = buffer.getvalue()

    # Base64 encode
    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return base64_string


def decode_image_base64(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy image.

    Args:
        base64_string: Base64-encoded image string

    Returns:
        Numpy array in BGR format

    Raises:
        ValueError: If decoding fails
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        image_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

        return image_array
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def resize_image(
    image: np.ndarray, max_width: int, max_height: int, maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to fit within dimensions.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if width <= max_width and height <= max_height:
        return image.copy()

    if maintain_aspect:
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = max_width
        new_height = max_height

    # Use appropriate interpolation
    if new_width < width:
        interpolation = cv2.INTER_AREA  # Best for downscaling
    else:
        interpolation = cv2.INTER_LINEAR  # Good for upscaling

    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return resized


def crop_image(image: np.ndarray, bbox: "BoundingBox") -> np.ndarray:
    """
    Crop image using bounding box.

    Args:
        image: Input image
        bbox: BoundingBox object with x, y, width, height

    Returns:
        Cropped image

    Raises:
        ValueError: If bbox is invalid or out of bounds
    """
    height, width = image.shape[:2]

    # Ensure bbox is within image bounds
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(width, bbox.x + bbox.width)
    y2 = min(height, bbox.y + bbox.height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: results in zero or negative area")

    cropped = image[y1:y2, x1:x2].copy()
    return cropped


def draw_bboxes(
    image: np.ndarray,
    bboxes: List["BoundingBox"],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes on image for visualization.

    Args:
        image: Input image
        bboxes: List of BoundingBox objects
        labels: Optional list of labels for each bbox
        colors: Optional list of BGR colors for each bbox
        thickness: Line thickness
        font_scale: Label font scale

    Returns:
        Image with drawn bounding boxes
    """
    result = image.copy()

    # Default colors (cycling through a palette)
    default_colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    for i, bbox in enumerate(bboxes):
        # Get color
        if colors and i < len(colors):
            color = colors[i]
        else:
            color = default_colors[i % len(default_colors)]

        # Draw rectangle
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]

            # Get label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw label background
            cv2.rectangle(
                result,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1,  # Filled
            )

            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
            )

    return result


def calculate_image_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Calculate perceptual hash of image (average hash algorithm).

    Args:
        image: Input image
        hash_size: Size of hash (default: 8 for 64-bit hash)

    Returns:
        Hash string (hexadecimal)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to hash_size x hash_size
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # Calculate mean
    mean = resized.mean()

    # Create binary hash
    diff = resized > mean

    # Convert to hexadecimal string
    hash_bytes = np.packbits(diff.flatten()).tobytes()
    hash_hex = hash_bytes.hex()

    return hash_hex


def images_are_similar(
    image1: np.ndarray, image2: np.ndarray, threshold: float = 0.9
) -> bool:
    """
    Check if two images are similar using perceptual hashing.

    Args:
        image1: First image
        image2: Second image
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        True if images are similar above threshold
    """
    hash1 = calculate_image_hash(image1)
    hash2 = calculate_image_hash(image2)

    # Calculate Hamming distance
    hash1_bits = bin(int(hash1, 16))[2:].zfill(64)
    hash2_bits = bin(int(hash2, 16))[2:].zfill(64)

    hamming_distance = sum(b1 != b2 for b1, b2 in zip(hash1_bits, hash2_bits))

    # Convert to similarity (0.0 = different, 1.0 = identical)
    similarity = 1.0 - (hamming_distance / 64.0)

    return similarity >= threshold


def calculate_md5_hash(image: np.ndarray) -> str:
    """
    Calculate MD5 hash of image data (exact match, not perceptual).

    Args:
        image: Input image

    Returns:
        MD5 hash string
    """
    image_bytes = image.tobytes()
    md5_hash = hashlib.md5(image_bytes).hexdigest()
    return md5_hash


def pad_image(
    image: np.ndarray, target_width: int, target_height: int, pad_value: int = 255
) -> np.ndarray:
    """
    Pad image to target size with specified value.

    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        pad_value: Padding value (default: 255 for white)

    Returns:
        Padded image
    """
    height, width = image.shape[:2]

    if width > target_width or height > target_height:
        raise ValueError(
            f"Image ({width}x{height}) larger than target ({target_width}x{target_height})"
        )

    # Calculate padding
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top

    # Pad image
    if len(image.shape) == 2:
        # Grayscale
        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    else:
        # Color
        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

    return padded


def create_thumbnail(image: np.ndarray, max_size: int = 256) -> np.ndarray:
    """
    Create thumbnail of image with max dimension.

    Args:
        image: Input image
        max_size: Maximum dimension (width or height)

    Returns:
        Thumbnail image
    """
    return resize_image(image, max_size, max_size, maintain_aspect=True)


def concatenate_images(
    images: List[np.ndarray],
    orientation: str = "horizontal",
    spacing: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Concatenate multiple images with spacing.

    Args:
        images: List of images to concatenate
        orientation: 'horizontal' or 'vertical'
        spacing: Spacing between images in pixels
        background_color: Background color for spacing (BGR)

    Returns:
        Concatenated image
    """
    if not images:
        raise ValueError("No images provided")

    if orientation == "horizontal":
        # Find max height
        max_height = max(img.shape[0] for img in images)

        # Resize all to same height
        resized = []
        for img in images:
            if img.shape[0] != max_height:
                scale = max_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img_resized = cv2.resize(img, (new_width, max_height))
                resized.append(img_resized)
            else:
                resized.append(img)

        # Calculate total width
        total_width = sum(img.shape[1] for img in resized)
        total_width += spacing * (len(resized) - 1)

        # Create canvas
        if len(images[0].shape) == 3:
            canvas = np.full(
                (max_height, total_width, images[0].shape[2]),
                background_color,
                dtype=np.uint8,
            )
        else:
            canvas = np.full(
                (max_height, total_width), background_color[0], dtype=np.uint8
            )

        # Place images
        x_offset = 0
        for img in resized:
            canvas[:, x_offset : x_offset + img.shape[1]] = img
            x_offset += img.shape[1] + spacing

        return canvas

    else:  # vertical
        # Find max width
        max_width = max(img.shape[1] for img in images)

        # Resize all to same width
        resized = []
        for img in images:
            if img.shape[1] != max_width:
                scale = max_width / img.shape[1]
                new_height = int(img.shape[0] * scale)
                img_resized = cv2.resize(img, (max_width, new_height))
                resized.append(img_resized)
            else:
                resized.append(img)

        # Calculate total height
        total_height = sum(img.shape[0] for img in resized)
        total_height += spacing * (len(resized) - 1)

        # Create canvas
        if len(images[0].shape) == 3:
            canvas = np.full(
                (total_height, max_width, images[0].shape[2]),
                background_color,
                dtype=np.uint8,
            )
        else:
            canvas = np.full(
                (total_height, max_width), background_color[0], dtype=np.uint8
            )

        # Place images
        y_offset = 0
        for img in resized:
            canvas[y_offset : y_offset + img.shape[0], :] = img
            y_offset += img.shape[0] + spacing

        return canvas

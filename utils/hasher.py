from PIL import Image
import hashlib
import base64
import imagehash


def generate_image_id(image: Image, size=(256, 256), hash_length=16):
    """Generate a shorter unique ID for an image by truncating the SHA256 hash."""
    try:
        img = image.resize(size).convert('RGB')
        pixel_data = img.tobytes()
        sha_hash = hashlib.sha256(pixel_data).hexdigest()
        return sha_hash[:hash_length]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    

def generate_md5_image_id(image: Image, size=(256, 256)):
    """Generate a shorter unique ID for an image using MD5 hash."""
    try:
        img = image.resize(size).convert('RGB')
        pixel_data = img.tobytes()
        md5_hash = hashlib.md5(pixel_data).hexdigest()
        return md5_hash
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def generate_base64_image_id(image: Image, size=(256, 256)):
    """Generate a shorter unique ID for an image using Base64 encoding."""
    try:
        img = image.resize(size).convert('RGB')
        pixel_data = img.tobytes()
        sha_hash_bytes = hashlib.sha256(pixel_data).digest()
        base64_hash = base64.urlsafe_b64encode(sha_hash_bytes).decode('utf-8').rstrip('=')
        return base64_hash
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def generate_short_perceptual_image_id(image: Image, hash_length=8):
    """Generate a short perceptual hash for an image."""
    try:
        perceptual_hash = imagehash.phash(image)
        return str(perceptual_hash)[:hash_length]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
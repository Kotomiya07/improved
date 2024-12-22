import os
import numpy as np
from PIL import Image
import argparse

def load_and_preprocess_images(path):
    """Load and preprocess images from the given path."""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Get all image files from the directory
    for filename in sorted(os.listdir(path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            # Load image
            img_path = os.path.join(path, filename)
            img = Image.open(img_path).convert('RGB')
            
            # Convert to numpy array and transpose to (C, H, W)
            img_array = np.array(img).transpose(2, 0, 1)
            images.append(img_array)
    
    # Stack all images into a single numpy array
    return np.stack(images)  # Result will be (N, C, H, W)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert images to numpy array')
    parser.add_argument('--image_path', type=str, help='Path to the directory containing images')
    parser.add_argument('--output', type=str, default='images.npy',
                        help='Output filename (default: images.npy)')
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Path {args.image_path} does not exist")
        return
    
    # Load and convert images
    print("Loading and converting images...")
    images = load_and_preprocess_images(args.image_path)
    
    # Save to npy file
    print(f"Saving {len(images)} images to {args.output}")
    np.save(args.output, images)
    print("Done!")
    print(f"Array shape: {images.shape}")

if __name__ == '__main__':
    main()
# corruption_dataset.py
import cv2
import numpy as np
import random
import os
import glob # Used to get all file paths

def apply_gaussian_blur(img):
    """Applies a random Gaussian blur to the image."""
    kernel_size = random.choice([3, 5, 7]) # Pick a random kernel size
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_salt_pepper_noise(img):
    """Applies salt and pepper noise."""
    row, col, _ = img.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)
    # Salt
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords[0], coords[1], :] = 255
    # Pepper
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[coords[0], coords[1], :] = 0
    return out

def apply_blackout_patch(img):
    """Applies the original random blackout patch."""
    rows, cols, _ = img.shape
    patch_size = 60
    x1 = np.random.randint(0, cols - patch_size)
    y1 = np.random.randint(0, rows - patch_size)
    corrupted_img = img.copy()
    corrupted_img[y1:y1 + patch_size, x1:x1 + patch_size, :] = 0
    return corrupted_img

def apply_jpeg_artifacts(img):
    """Simulates JPEG compression artifacts."""
    # Encode with low quality and then decode
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(5, 20)])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

def corrupt_image_randomly(image_path, save_path):
    """
    Loads an image and applies one of the various corruption techniques at random.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} not found.")
        return

    # List of all possible corruption functions
    corruption_functions = [
        apply_gaussian_blur,
        apply_salt_pepper_noise,
        apply_blackout_patch,
        apply_jpeg_artifacts
    ]

    # 1. Randomly choose one of the corruption functions
    chosen_corruption = random.choice(corruption_functions)
    
    # 2. Apply it to the image
    corrupted_img = chosen_corruption(img)
    
    # 3. You can even combine them for more complex damage
    # For example, let's sometimes add a second, different corruption
    if random.random() > 0.5: # 50% chance to add another corruption
        # Remove the already chosen function to avoid applying it twice
        corruption_functions.remove(chosen_corruption)
        second_corruption = random.choice(corruption_functions)
        corrupted_img = second_corruption(corrupted_img)

    cv2.imwrite(save_path, corrupted_img)

# --- Main script to process the entire folder ---
CLEAN_DIR = "scraped_images"
CORRUPTED_DIR = "corrupted_images"
VERSIONS_PER_IMAGE = 5

if not os.path.exists(CORRUPTED_DIR):
    os.makedirs(CORRUPTED_DIR)

clean_image_paths = glob.glob(os.path.join(CLEAN_DIR, "*.jpg"))
print(f"Found {len(clean_image_paths)} clean images.")
print(f"Creating {VERSIONS_PER_IMAGE} corrupted versions for each...")

for i, image_path in enumerate(clean_image_paths):
    # Get the original filename without the extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- This is the new data augmentation loop ---
    for v in range(VERSIONS_PER_IMAGE):
        # Create a new filename for each version, e.g., "my_photo_v1.jpg"
        new_filename = f"{base_filename}_v{v+1}.jpg"
        save_path = os.path.join(CORRUPTED_DIR, new_filename)
        
        # Call the random corruption function
        corrupt_image_randomly(image_path, save_path)
    
    print(f"({i+1}/{len(clean_image_paths)}) Created {VERSIONS_PER_IMAGE} versions for '{os.path.basename(image_path)}'")

print(f"\nData augmentation complete. Total corrupted images: {len(clean_image_paths) * VERSIONS_PER_IMAGE}")



















# # Create the output directory if it doesn't exist
# if not os.path.exists(CORRUPTED_DIR):
#     os.makedirs(CORRUPTED_DIR)

# # Get a list of all image files in the clean directory
# # glob.glob finds all pathnames matching a specified pattern
# clean_image_paths = glob.glob(os.path.join(CLEAN_DIR, "*.jpg"))
# print(f"Found {len(clean_image_paths)} images to corrupt.")

# # Loop through each clean image path
# for i, image_path in enumerate(clean_image_paths):
#     # Get the original filename to use for the corrupted version
#     filename = os.path.basename(image_path)
#     save_path = os.path.join(CORRUPTED_DIR, filename)
    
#     # Call the corruption function
#     corrupt_image_randomly(image_path, save_path)
    
#     # Print progress
#     print(f"({i+1}/{len(clean_image_paths)}) Corrupted '{filename}'")

# print("\nCorruption process complete.")
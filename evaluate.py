# evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# Import your model and dataset classes from your training script
# Make sure train_gan.py is in the same folder or accessible
from train_gan import Generator, ImageRestorationDataset

## 1. CONFIGURATION
# -------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # Process one image at a time for evaluation
IMAGE_SIZE = 256

# --- IMPORTANT: UPDATE THIS PATH ---
# Path to your saved generator model from training
GEN_MODEL_PATH = "saved_models/gen_epoch_499.pth" 

# Test data folders
ROOT_CLEAN_TEST = "test_set/clean"
ROOT_CORRUPTED_TEST = "test_set/corrupted"

## 2. LOAD MODEL AND DATA
# -------------------------------------------
print("Loading model and data...")
# Load the generator
gen = Generator(in_channels=3).to(DEVICE)
gen.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE))
gen.eval() # Set model to evaluation mode

# Load the test dataset
test_dataset = ImageRestorationDataset(root_corrupted=ROOT_CORRUPTED_TEST, root_clean=ROOT_CLEAN_TEST)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the LPIPS model
loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)

## 3. EVALUATION LOOP
# -------------------------------------------
psnr_scores = []
ssim_scores = []
lpips_scores = []

print("Starting evaluation...")
loop = tqdm(test_loader, leave=True)

# Helper function to convert tensor to numpy array for skimage
def tensor_to_numpy(tensor):
    # De-normalize from [-1, 1] to [0, 1] and then to [0, 255]
    img = tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
    img = (img + 1) / 2
    img_numpy = (img.numpy() * 255).astype(np.uint8)
    return img_numpy

for corrupted_img_tensor, clean_img_tensor in loop:
    corrupted_img_tensor = corrupted_img_tensor.to(DEVICE)
    clean_img_tensor = clean_img_tensor.to(DEVICE)
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        restored_img_tensor = gen(corrupted_img_tensor)

    # --- Calculate metrics ---
    # Convert tensors to format suitable for metric calculation
    clean_np = tensor_to_numpy(clean_img_tensor)
    restored_np = tensor_to_numpy(restored_img_tensor)
    
    # 1. PSNR
    current_psnr = psnr(clean_np, restored_np, data_range=255)
    psnr_scores.append(current_psnr)
    
    # 2. SSIM
    # For multichannel (RGB), specify multichannel=True
    current_ssim = ssim(clean_np, restored_np, data_range=255, multichannel=True, channel_axis=2)
    ssim_scores.append(current_ssim)
    
    # 3. LPIPS
    # LPIPS function expects tensors in range [-1, 1], which is what our model outputs
    current_lpips = loss_fn_lpips(restored_img_tensor, clean_img_tensor)
    lpips_scores.append(current_lpips.item())
    
    loop.set_postfix(
        PSNR=f"{np.mean(psnr_scores):.2f}",
        SSIM=f"{np.mean(ssim_scores):.4f}",
        LPIPS=f"{np.mean(lpips_scores):.4f}"
    )

## 4. REPORT FINAL RESULTS
# -------------------------------------------
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)
avg_lpips = np.mean(lpips_scores)

print("\n--- Evaluation Complete ---")
print(f"Number of test images: {len(test_dataset)}")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")
print("---------------------------")
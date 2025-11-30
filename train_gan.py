# train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

import os
from PIL import Image
from tqdm import tqdm

## 1. CONFIGURATION / HYPERPARAMETERS
# -------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 13
NUM_EPOCHS = 500
IMAGE_SIZE = 256 # The size to resize images to
CHANNELS_IMG = 3
L1_LAMBDA = 100 # Weight for the L1 reconstruction loss

# Data folders
ROOT_CLEAN = "scraped_images"
ROOT_CORRUPTED = "corrupted_images"
# Output folders
SAVE_MODEL_DIR = "saved_models"
SAVE_IMAGE_DIR = "saved_images"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

## 2. CUSTOM DATASET
# -------------------------------------------
# Loads pairs of (corrupted, clean) images
class ImageRestorationDataset(Dataset):
    def __init__(self, root_corrupted, root_clean):
        self.root_corrupted = root_corrupted
        self.root_clean = root_clean
        
        self.corrupted_files = sorted(os.listdir(self.root_corrupted))
        self.clean_files_map = {f: f for f in os.listdir(self.root_clean)}

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.corrupted_files)

    def __getitem__(self, index):
        corrupted_file = self.corrupted_files[index]
        corrupted_path = os.path.join(self.root_corrupted, corrupted_file)
        
        # Find the corresponding clean file (e.g., "image1_v1.jpg" -> "image1.jpg")
        base_name = "_".join(corrupted_file.split('_')[:-1]) + ".jpg"
        clean_path = os.path.join(self.root_clean, base_name)
        
        corrupted_img = Image.open(corrupted_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        corrupted_img = self.transform(corrupted_img)
        clean_img = self.transform(clean_img)
        
        return corrupted_img, clean_img

## 3. MODEL ARCHITECTURE (Pix2Pix GAN)
# -------------------------------------------
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), # 1. Upsample
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # 2. Convolve
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"), nn.LeakyReLU(0.2))
        self.down1 = Block(features, features * 2, act="leaky")
        self.down2 = Block(features * 2, features * 4, act="leaky")
        self.down3 = Block(features * 4, features * 8, act="leaky")
        self.down4 = Block(features * 8, features * 8, act="leaky")
        self.down5 = Block(features * 8, features * 8, act="leaky")
        self.down6 = Block(features * 8, features * 8, act="leaky")
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), nn.ReLU())
        self.up1 = Block(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False)
        self.up7 = Block(features * 2 * 2, features, down=False)
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features * 2, in_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(self.initial(x))


## 4. MAIN TRAINING SCRIPT
# -------------------------------------------
if __name__ == "__main__":
    # -- Initialize models --
    disc = Discriminator(in_channels=CHANNELS_IMG).to(DEVICE)
    gen = Generator(in_channels=CHANNELS_IMG).to(DEVICE)

    # -- Initialize optimizers --
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # -- Loss Functions --
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()
    
    # -- Data Loader --
    dataset = ImageRestorationDataset(root_corrupted=ROOT_CORRUPTED, root_clean=ROOT_CLEAN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print("🚀 Starting Training...")
    
    # -- Training Loop --
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (corrupted, clean) in enumerate(loop):
            corrupted = corrupted.to(DEVICE)
            clean = clean.to(DEVICE)
            
            # --- Train Discriminator ---
            # 1. Generate a fake image
            fake = gen(corrupted)
            
            # 2. Train on real images
            D_real = disc(corrupted, clean)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            
            # 3. Train on fake images
            D_fake = disc(corrupted, fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            
            # 4. Combine losses and update
            D_loss = (D_real_loss + D_fake_loss) / 2
            
            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            # 1. Adversarial loss (try to fool the discriminator)
            D_fake = disc(corrupted, fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            
            # 2. Reconstruction loss (L1)
            L1_loss = L1(fake, clean) * L1_LAMBDA
            
            # 3. Combine losses and update
            G_loss = G_fake_loss + L1_loss
            
            gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            # Update progress bar
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )

        # -- Save a grid of images to check progress --
        # Take the first 8 images from the last batch
        img_grid = torch.cat((corrupted[:8] * 0.5 + 0.5, fake[:8] * 0.5 + 0.5, clean[:8] * 0.5 + 0.5))
        save_image(img_grid, os.path.join(SAVE_IMAGE_DIR, f"epoch_{epoch}.png"), nrow=8)
        
        # -- Save model checkpoints --
        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), os.path.join(SAVE_MODEL_DIR, f"gen_epoch_{epoch}.pth"))
            torch.save(disc.state_dict(), os.path.join(SAVE_MODEL_DIR, f"disc_epoch_{epoch}.pth"))
            print(f"💾 Checkpoint saved at epoch {epoch}")

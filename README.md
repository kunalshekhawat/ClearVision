# ClearVision: High-Fidelity Image Restoration GAN
ClearVision is an end-to-end deep learning pipeline designed to perform high-fidelity image inpainting and denoising on corrupted visual data. Built from scratch in PyTorch, this project implements a customized **Pix2Pix Conditional GAN (cGAN)** to restore images degraded by missing pixels, Gaussian blur, and salt-and-pepper noise.

## 🚀 Key Features

* **Custom Data Pipeline:** Scraped a raw image dataset using Selenium, followed by automated programmatic corruption (blur, noise, masking) to generate paired training data.
* **Architectural Improvements:** Eliminated native GAN "checkerboard artifacts" by replacing standard Transposed Convolutions with a custom Resize-Convolution approach (Bilinear Upsampling + Conv2d) in the U-Net Generator.
* **PatchGAN Discriminator:** Utilized a patch-level discriminator to enforce high-frequency structural details and prevent blurry hallucinated outputs.
* **Real-time Inference:** Deployed the trained model as an interactive web application using Streamlit for seamless, real-time visual demonstration.

## 🧠 Model Architecture

The core architecture is based on the Pix2Pix framework, utilizing a combined adversarial and reconstruction loss function: `Total Loss = cGAN Loss + (λ * L1 Loss)`.

* **Generator (Modified U-Net):** An encoder-decoder network with skip connections to preserve low-level spatial details. The upsampling layers were custom-engineered to prevent uneven pixel overlap during image reconstruction.
* **Discriminator (PatchGAN):** Evaluates $N \times N$ patches of the image to determine authenticity, forcing the generator to produce sharp, realistic textures rather than just minimizing global pixel errors.

## 📊 Quantitative Results

The model was evaluated on a dedicated, unseen test set and achieved highly competitive restoration metrics:

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **PSNR** | `25.15 dB` | High peak signal-to-noise ratio, indicating low mean squared error at the pixel level. |
| **SSIM** | `0.7987` | Strong structural similarity index, proving the structural integrity of the restored image. |
| **LPIPS** | `0.2209` | Low learned perceptual distance, validating that outputs are highly realistic to the human eye. |

## 📂 Project Structure

```text
ClearVision/
│
├── scraper.py          # Selenium script for web-scraping training images
├── corruption.py       # Applies Gaussian blur, noise, and masks to create paired data
├── train_gan.py        # PyTorch model definitions and main training loop
├── evaluate.py         # Script to calculate PSNR, SSIM, and LPIPS metrics
├── app.py              # Streamlit web application for real-time inference
├── clearvision.ipynb   # Jupyter Notebook containing exploratory data analysis & training
└── README.md           # Project documentation


⚙️ Installation & Usage
1. Clone the repository:
git clone [https://github.com/kunalshekhawat/ClearVision.git](https://github.com/kunalshekhawat/ClearVision.git)
cd ClearVision

2. Install dependencies:
pip install torch torchvision torchaudio streamlit selenium Pillow tqdm

3. Run the Web Application:
To test the pre-trained model on your own corrupted images, launch the Streamlit app:
streamlit run app.py

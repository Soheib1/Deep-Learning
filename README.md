# Miniproject : Image Generation with a Diffusion Model

## Overview
This project involves building a **diffusion model** from scratch to generate images. Diffusion models are a class of generative models that iteratively refine images from pure noise to resemble real data. The implementation is inspired by the paper **"Elucidating the Design Space of Diffusion-Based Generative Models"** by Tero Karras et al., presented at **NeurIPS 2022**.  

📄 **Paper Link:** [https://openreview.net/pdf?id=k7FuTOWMOc7](https://openreview.net/pdf?id=k7FuTOWMOc7)  

The results are mainly in the "projet.ipynb" file

---

## How It Works
The project is based on two key processes:

1. **Noising Process**  
   - Starts from clean images.  
   - Gradually adds noise until the images become indistinguishable from pure noise.

2. **Denoising Process**  
   - Starts from pure noise.  
   - Iteratively removes noise using a trained neural network to reconstruct realistic images.

---

## Implementation Details
- The project is implemented in **Python** using **PyTorch**.  
- The model learns to reverse the noising process to generate new images from random noise.  
- Key components include:  
  -  **Forward Diffusion Process**: Gradually corrupts an image with Gaussian noise.  
  -  **Neural Network Training**: Learns to denoise the corrupted images.  
  -  **Sampling Process**: Generates images by iteratively denoising random noise.  

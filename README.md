# kt-Dynamic-MRI-Reconstruction

This is a reference implementation for the following work:

[MICCAI'19] k-t NEXT: Dynamic MR Image Reconstruction Exploiting Spatio-Temporal Correlations. 
Paper link: https://arxiv.org/abs/1907.09425

[Pending] Complementary Time-Frequency Domain Network for Dynamic Parallel MR Image Reconstruction.

## Introduction

Dynamic magnetic resonance imaging (MRI) exhibits high correlations in k-space and time. In order to accelerate the dynamic MR imaging and to exploit k-t correlations from highly undersampled data, here we develop a novel deep learning based approach for dynamic MR image reconstruction, termed k-t NEXT (k-t NEtwork with X-f Transform). In particular, we reconstruct the true signals from aliased signals in x-f domain to exploit the spatio-temporal redundancies. Building on that, the proposed method then learns to recover the signals by alternating the reconstruction process between the x-f space and image space in an iterative fashion. This enables the network to effectively capture useful information and jointly exploit spatio-temporal correlations from both complementary domains. 

This repository contains the implementation of xf-CNN and CRNN-MRI using PyTorch, along with a simple demo. PyTorch version needs to be higher than Torch 0.4.

## Usage

  python main_kt_NEXT.py --acceleration_factor 4
  
## Citation and Acknowledgement

If you use the code for your work, or if you found the code useful, please cite the following work.

Qin C. et al. (2019) k-t NEXT: Dynamic MR Image Reconstruction Exploiting Spatio-Temporal Correlations. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11765. Springer, Cham. https://doi.org/10.1007/978-3-030-32245-8_56

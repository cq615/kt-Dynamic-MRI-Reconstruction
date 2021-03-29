# kt-Dynamic-MRI-Reconstruction

This is a reference implementation for the following work:

[MICCAI'19] k-t NEXT: Dynamic MR Image Reconstruction Exploiting Spatio-Temporal Correlations. 

Paper link: https://arxiv.org/abs/1907.09425

[Pending] Complementary Time-Frequency Domain Network for Dynamic Parallel MR Image Reconstruction.

Paper link: https://arxiv.org/abs/2012.11974

## Introduction

Dynamic magnetic resonance imaging (MRI) exhibits high correlations in k-space and time. In order to accelerate the dynamic MR imaging and to exploit k-t correlations from highly undersampled data, here we develop novel deep learning based approaches for dynamic MR image reconstruction in both single-coil and multi-coil acquisition settings. The developed approaches are termed k-t NEXT (k-t NEtwork with X-f Transform) for sinlge-coil setting and CTFNet (Complementary Time-Frequency domain Network) for multi-coil setting. The networks are able to effectively capture useful information and jointly exploit spatio-temporal correlations from both complementary domains (x-f and image domains). 

This repository contains the implementation of k-t NEXT and CTFNet including the CRNN-MRI module using PyTorch, along with a simple demo. PyTorch version needs to be higher than Torch 0.4.

## Usage

To train k-t NEXT on single-coil data:

    python main_kt_NEXT.py --acceleration_factor 4
  
To train CTFNet on multi-coil data:

    python main_CTFNet.py --acceleration_factor 8
  
## Citation and Acknowledgement

If you use the code for your work, or if you found the code useful, please cite the following work.

Qin C, et al. (2019) k-t NEXT: Dynamic MR Image Reconstruction Exploiting Spatio-Temporal Correlations. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11765. Springer, Cham. https://doi.org/10.1007/978-3-030-32245-8_56

Qin C, et al. (2020) "Complementary Time-Frequency Domain Networks for Dynamic Parallel MR Image Reconstruction." arXiv preprint arXiv:2012.11974.

If you use the CRNN-MRI module, please also cite:

C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.

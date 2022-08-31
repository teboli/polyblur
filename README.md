# Pytorch and Numpy Non-official Implementation of Polyblur

| <img src="./peacock_defocus.png" width="360px"/> | <img src="results/peacock_restored_alpha_6_beta_1.png" width="360px"/> |
|:------------------------------------------------:|:----------------------------------------------------------------------:|
|        <i>Slightly out-of-focus image</i>        |                 <i>Deblurred result with Polyblur</i>                  |

This repository contains a Python implementation of *Polyblur: Removing mild blur by polynomial reblurring* in
IEEE Transaction on Computational Imaging 2021 by Mauricio Delbracio, Ignacio Garcia-Dorado, Soungjoon Choi, 
Damien Kelly and Peyman Milanfar. We provide Pytorch and Numpy non-official implementations reproducing the quantitative
and qualitative results of the original paper.

A description an analysis of the algorithm can be found in a companion IPOL paper:
*Breaking down Polyblur: Fast blind Correction of Small Anisotropic Blurs* in Image Processing OnLine 2022 by Thomas Eboli, 
Jean-Michel Morel and Gabriele Facciolo.

### Test

First install the requirements with
> pip install requirements.txt

The Pytorch implementation of this code runs **ONLY** with torch 1.10+ 
versions (because of torch.fft for the computation of the gradients with the pytorch implementation).

Once done, you can test the blind deblurring technique with
> python main.py --impath peacock_defocus.png --N 3 --alpha 6 --beta 1

You can modify several parameters, e.g. the number of Polyblur iterations and
the deconvolution filter's parameters alpha and beta


### Description

The code contains *functional* and *module* blocks: The first is found in the path
> deblurring.polyblur

and is agnostic to (H,W,C) or (H,W) Numpy arrays or (B,C,H,W) Pytorch tensors. The latter inherits from torch.nn.module 
and either call deblurring.polyblur on the whole image or overlapping patches. This module is called
with 
> deblurring.Polyblur



### Calibration

We also provide a code for reproducing the calibration curves of the original paper with our FFT-based implementation of 
the gradients, or any other one one can image. To compute the slope and intercept 
of the model, please run
> python calibration_blur_parameters.py

Below is an example of the predicted curves for 1% additive Gaussian noise. If you use the exact same implementation 
of the gradients as in this repo, you should find something  like (0.362, -0.468) for the affine model parameters.
You should first provide the path
to the DVI2K validation set.

| <img src="./results/calibration_normal_0.01.jpg" width="360px"/> | <img src="results/calibration_orthogonal_0.01.jpg" width="360px"/> |
|:----------------------------------------------------------------:|:------------------------------------------------------------------:|
|                 <i>Principal blur direction</i>                  |                  <i>Orthogonal blur direction</i>                  |



### Contact 

If you encounter any problem with the code, please contact me at <thomas.eboli@ens-paris-saclay.fr>.


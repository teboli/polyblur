from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .import utils
from .filters import gaussian_filter, fourier_gradients


##############################################
############ Pytorch Sub-routines ############
##############################################



def gaussian_blur_estimation(imgc, q=0.0001, n_angles=6, n_interpolated_angles=30, c=0.362, b=0.464, ker_size=25,
                                   discard_saturation=False, multichannel=False, thetas=None, interpolated_thetas=None,
                                   return_2d_filters=True):
    """
    Compute an approximate Gaussian filter from a RGB or grayscale image.
        :param img: (B,C,H,W) or (B,1,H,W) tensor, the image
        :param q: the quantile value used for image normalization
        :param n_angles: the number of angles to compute the directional gradients
        :param n_interpolated_angles: the number of angles to interpolate the directional gradients
        :param c: the slope of the affine model
        :param b: the intercept of the affine model
        :param ker_size: the size of the kernel support
        :param discard_saturation: taking care of saturated areas in gradient computation
        :param multichannel: predicting a kernel per channel or on grayscale image
        :param q: the quantile for image normalization (typically 0 or 1e-4)
        :return: kernel: the (B,C,ker_size,ker_sizr) or (B,1,ker_size,ker_size) local Gaussian kernels
    """
    # if img is color or multichannel is False (same kernel for each color channel), go to grayscale
    if imgc.shape[1] == 3 or not multichannel:
        imgc = imgc.mean(dim=1, keepdims=True)  # BxCxHxW becomes Bx1xHxW

    # init
    if thetas is None:
        thetas = torch.linspace(0, 180, n_angles+1).unsqueeze(0)   # (1,n)
        if torch.cuda.is_available():
            thetas = thetas.to(imgc.device)
    if interpolated_thetas is None:
        interpolated_thetas = torch.arange(0, 180, 180 / n_interpolated_angles).unsqueeze(0)   # (1,N)
        if torch.cuda.is_available():
            interpolated_thetas = interpolated_thetas.to(imgc.device)

    # kernel estimation
    if return_2d_filters:
        kernel = torch.zeros(*imgc.shape[:2], ker_size, ker_size, device=imgc.device).float()  # BxCxhxw or Bx1xhxw
    else:
        kernel = (torch.zeros(*imgc.shape[:2], device=imgc.device),  # sigmas
                  torch.zeros(*imgc.shape[:2], device=imgc.device),  # rhos
                  torch.zeros(*imgc.shape[:2], device=imgc.device))  # thetas
    for channel in range(imgc.shape[1]):
        img = imgc[:, channel:channel+1]  # Bx1xHxW
        # (Optional) remove saturated areas
        mask = get_saturation_mask(img, discard_saturation)
        # normalized image
        img_normalized = normalize(img, q=q)
        # compute the image gradients
        gradients = compute_gradients(img_normalized, mask=mask)
        # compute the gradient magnitudes per orientation
        gradient_magnitudes = compute_gradient_magnitudes(gradients, n_angles=n_angles)
        # find the maximal blur direction amongst sampled orientations
        magnitude_normal, magnitude_ortho, thetas = find_maximal_blur_direction(gradient_magnitudes, 
                                                                                thetas, interpolated_thetas)
        # finally compute the Gaussian parameters
        sigma, rho = compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=c, b=b)
        # create the blur kernel or store the Gaussian parameters
        if return_2d_filters:
            kernel[:, channel:channel+1] = create_gaussian_filter(thetas, sigma, rho, ksize=ker_size)
        else:
            kernel[0][:,channel:channel+1] = sigma
            kernel[1][:,channel:channel+1] = rho
            kernel[2][:,channel:channel+1] = theta
        
    return kernel


# @torch.jit.script
def get_saturation_mask(img, discard_saturation, threshold=0.99):
    if discard_saturation:
        mask = img > threshold
    else:
        mask = img > 1  # every entry is False
    return mask 


# @torch.jit.script
def clamp_(img, max=1.0, min=0.0):
    return ( (img - min) / (max - min) ).clamp(0.0, 1.0)


def normalize(images, q=0.0001):
    """
    range normalization of the images by clipping a small quantile
    """
    # Pytorch quantile are *much* slower than computing max/min but prevent 
    # wrong predictions in the presence of artifacts or noise.
    if q > 0:
        b, c, h, w = images.shape
        value_min = torch.quantile(images.view(b, c, -1), q=q, dim=-1, keepdim=True)
        value_max = torch.quantile(images.view(b, c, -1), q=1-q, dim=-1, keepdim=True)
    else:
        value_min = torch.amin(images, dim=(-1,-2), keepdim=True)
        value_max = torch.amax(images, dim=(-1,-2), keepdim=True)
    return clamp_(images, value_max, value_min)


def compute_gradients(img, mask):
    """
    compute fourier gradients
    """
    gradient_x, gradient_y = fourier_gradients(img)
    gradient_x[mask] = 0
    gradient_y[mask] = 0
    return gradient_x, gradient_y


def compute_gradient_magnitudes(gradients, n_angles=6):
    """
    compute the maximum of the gradient magnitudes for each angle 
    """
    gradient_x, gradient_y = gradients  # (B,C,H,W)
    gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    angles = torch.linspace(0, np.pi, n_angles + 1, device=gradient_x.device).view(1, -1, 1, 1, 1)  # (1,N,1,1,1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
    gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
    return gradient_magnitudes_angles


@torch.jit.script
def cubic_interpolator(x_new, x, y):
    """
    Fast implement of cubic interpolator based on Keys' algorithm
    """
    x_new = torch.abs(x_new[..., None] - x[..., None, :])
    mask1 = x_new < 1
    mask2 = torch.bitwise_and(1 <= x_new, x_new < 2)
    x_new = mask2.float() * (((-0.5 * x_new + 2.5) * x_new - 4) * x_new + 2) + \
            mask1.float() * ((1.5 * x_new - 2.5) * x_new * x_new + 1) 
    x_new /= torch.sum(x_new, dim=-1, keepdim=True) + 1e-5
    return (x_new @ y[..., None]).squeeze(-1)


def find_maximal_blur_direction(gradient_magnitudes_angles, thetas, interpolated_thetas):
    """
    Predict the blur's main direction by evaluating the maximum of the directional derivatives
    """
    ## Find thetas
    n_interpolated_angles = interpolated_thetas.shape[-1]
    gradient_magnitudes_interpolated_angles = cubic_interpolator(interpolated_thetas / n_interpolated_angles, 
                                                thetas / n_interpolated_angles, gradient_magnitudes_angles)  # (B,N)
    ## Compute magnitude in theta
    i_min = torch.argmin(gradient_magnitudes_interpolated_angles, dim=-1, keepdim=True)
    thetas_normal = torch.take_along_dim(interpolated_thetas, i_min, dim=-1)
    magnitudes_normal = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_min, dim=-1)
    ## Compute magnitude in theta+90
    thetas_ortho = (thetas_normal + 90) % 180  # angle in [0,180)
    i_ortho = (thetas_ortho / (180 / n_interpolated_angles)).long()
    magnitudes_ortho = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_ortho, dim=-1)
    return magnitudes_normal, magnitudes_ortho, thetas_normal.float() * np.pi / 180


# @ torch.jit.script
def compute_gaussian_parameters(magnitudes_normal, magnitudes_ortho, c, b):
    """
    Estimate the blur's standard deviations applying the affine model Eq.(24) followed by clipping
    """
    cc = c * c
    bb = b * b
    ## Compute sigma
    sigma = cc / (magnitudes_normal * magnitudes_normal + 1e-8) - bb
    sigma = torch.clamp(sigma, min=0.09, max=16.0)
    sigma = torch.sqrt(sigma)
    ## Compute rho
    rho = cc / (magnitudes_ortho * magnitudes_ortho + 1e-8) - bb
    rho = torch.clamp(rho, min=0.09, max=16.0)
    rho = torch.sqrt(rho)
    return sigma, rho


@torch.jit.script
def compute_gaussian_filter_parameters(sigmas, rhos, thetas):
    B = len(sigmas)
    C = 1
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = sigmas
    lambda_2 = rhos
    thetas = -thetas

    # Set COV matrix using Lambdas and Theta
    c = torch.cos(thetas)
    s = torch.sin(thetas)
    cc = c*c
    ss = s*s
    sc = s*c
    inv_lambda_1 = 1.0 / (lambda_1 * lambda_1)
    inv_lambda_2 = 1.0 / (lambda_2 * lambda_2)
    inv_sigma00 = cc * inv_lambda_1 + ss * inv_lambda_2
    inv_sigma01 = sc * (inv_lambda_1 - inv_lambda_2)
    inv_sigma11 = cc * inv_lambda_2 + ss * inv_lambda_1
    return inv_sigma00, inv_sigma01, inv_sigma11


def create_gaussian_filter(thetas, sigmas, rhos, ksize):
    """
    Outputs the generalized 2D gaussian kernels (of size ksize) determined by the eigenvalues thetas, sigmas, and angles rhos
    """
    B = sigmas.shape[0]
    C = 1

    # Create the inverse of the covariance matrix
    INV_SIGMA00, INV_SIGMA01, INV_SIGMA11 = compute_gaussian_filter_parameters(sigmas, rhos, thetas)
    INV_SIGMA = torch.stack([torch.stack([INV_SIGMA00, INV_SIGMA01], dim=-1),
                             torch.stack([INV_SIGMA01, INV_SIGMA11], dim=-1)], dim=-2)
    INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)

    # Create meshgrid for Gaussian
    t = torch.arange(ksize, device=sigmas.device) - ((ksize-1) // 2)
    X, Y = torch.meshgrid(t, t, indexing='xy')
    Z = torch.stack([X, Y], dim=-1).unsqueeze(-1).float()  # (k,k,2,1)
    Z_t = Z.transpose(-2, -1)  # (k,k,1,2)

    # Calculate Gaussian for every pixel of the kernel
    kernels = torch.exp(-0.5 * (Z_t @ INV_SIGMA @ Z)).view(B, C, ksize, ksize)  # (B,C,k,k)
    return kernels / torch.sum(kernels, dim=(-1,-2), keepdim=True)


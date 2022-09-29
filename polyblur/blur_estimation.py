import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .import utils
from .filters import gaussian_filter, fourier_gradients

from time import time


##############################################
############ Pytorch Sub-routines ############
##############################################



def gaussian_blur_estimation(imgc, q=0.0001, n_angles=6, n_interpolated_angles=30, c=0.362, b=0.464, ker_size=25,
                                   discard_saturation=False, multichannel=False, thetas=None, interpolated_thetas=None):
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
            thetas = thetas.to(imgc.device, non_blocking=True)
    if interpolated_thetas is None:
        interpolated_thetas = torch.arange(0, 180, 180 / n_interpolated_angles).unsqueeze(0)   # (1,N)
        if torch.cuda.is_available():
            interpolated_thetas = interpolated_thetas.to(imgc.device, non_blocking=True)

    start = time()
    print('    --init thetas:         %1.4f' %( time()-start ))

    # kernel estimation
    kernel = torch.zeros(*imgc.shape[:2], ker_size, ker_size, device=imgc.device).float()  # BxCxhxw or Bx1xhxw
    for channel in range(imgc.shape[1]):
        img = imgc[:, channel:channel+1]  # Bx1xHxW
        # (Optional) remove saturated areas
        start = time()
        mask = get_saturation_mask(img, discard_saturation)
        print('    --saturation:          %1.4f' % (time() - start))
        # normalized image
        start = time()
        img_normalized = normalize(img, q=q)
        print('    --normalize:           %1.4f' % (time() - start))
        # compute the image gradients
        start = time()
        gradients = compute_gradients(img_normalized, mask=mask)
        print('    --compute gradients:   %1.4f' % (time() - start))
        # compute the gradient magnitudes per orientation
        start = time()
        gradient_magnitudes = compute_gradient_magnitudes(gradients, n_angles=n_angles)
        print('    --compute magnitudes:  %1.4f' % (time() - start))
        # find the maximal blur direction amongst sampled orientations
        start = time()
        magnitude_normal, magnitude_ortho, thetas = find_maximal_blur_direction(gradient_magnitudes, 
                                                                                thetas, interpolated_thetas)
        print('    --find directions:     %1.4f' % (time() - start))
        # finally compute the Gaussian parameters
        start = time()
        sigma, rho = compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=c, b=b)
        print('    --gaussian parameter:  %1.4f' % (time() - start))
        # create the blur kernel
        start = time()
        k = create_gaussian_filter(thetas, sigma, rho, ksize=ker_size)
        print('    --create kernel:       %1.4f' % (time() - start))
        start = time()
        kernel[:, channel:channel+1] = k 
        print('    --store kernel:       %1.4f' % (time() - start))
        
    return kernel


def get_saturation_mask(img, discard_saturation, threshold=0.99):
    if discard_saturation:
        mask = img > threshold
    else:
        mask = img > 1  # every entry is False
    return mask 


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
    images = (images - value_min) / (value_max - value_min)
    return images.clamp(0.0, 1.0)


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


def cubic_interpolator(x_new, x, y):
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
    start = time()
    n_interpolated_angles = interpolated_thetas.shape[-1]
    gradient_magnitudes_interpolated_angles = cubic_interpolator(interpolated_thetas / n_interpolated_angles, 
                                                thetas / n_interpolated_angles, gradient_magnitudes_angles)  # (B,N)
    print('           -- finding thetas: %1.4f' % (time() - start))
    ## Compute magnitude in theta
    start = time()
    i_min = torch.argmin(gradient_magnitudes_interpolated_angles, dim=-1, keepdim=True).long()
    thetas_normal = torch.take_along_dim(interpolated_thetas, i_min, dim=-1)
    magnitudes_normal = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_min, dim=-1)
    print('           -- mag normal:     %1.4f' % (time() - start))
    ## Compute magnitude in theta+90
    start = time()
    thetas_ortho = (thetas_normal + 90.0) % 180  # angle in [0,pi)
    i_ortho = (thetas_ortho / (180 / n_interpolated_angles)).long()
    magnitudes_ortho = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_ortho, dim=-1)
    print('           -- mag ortho:      %1.4f' % (time() - start))
    return magnitudes_normal, magnitudes_ortho, thetas_normal * np.pi / 180


def compute_gaussian_parameters(magnitudes_normal, magnitudes_ortho, c, b):
    """
    Estimate the blur's standard deviations applying the affine model Eq.(24) followed by clipping
    """
    ## Compute sigma
    sigma = c**2 / (magnitudes_normal ** 2 + 1e-8) - b**2
    sigma = torch.maximum(sigma, 0.09 * torch.ones_like(sigma))
    sigma = torch.sqrt(sigma).clamp(0.3, 4.0)
    ## Compute rho
    rho = c**2 / (magnitudes_ortho ** 2 + 1e-8) - b**2
    rho = torch.maximum(sigma, 0.09 * torch.ones_like(rho))
    rho = torch.sqrt(rho).clamp(0.3, 4.0)
    return sigma, rho


def create_gaussian_filter(thetas, sigmas, rhos, ksize):
    """
    Outputs the generalized 2D gaussian kernels (of size ksize) determined by the eigenvalues thetas, sigmas, and angles rhos
    """
    B = len(sigmas)
    C = 1
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = sigmas
    lambda_2 = rhos
    thetas = -thetas

    # Create meshgrid for Gaussian
    start = time()
    t = torch.arange(ksize, dtype=torch.float32).to(sigmas.device, non_blocking=True)
    X, Y = torch.meshgrid(t, t, indexing='xy')
    Z = torch.stack([X, Y], dim=-1).unsqueeze(-1).float()  # (k,k,2,1)
    print('           --Z:         %1.4f' % (time() - start))
    start = time()
    
    # Set COV matrix using Lambdas and Theta
    #  # Sigma = Q LAMBDA Qt thus inv_Sigma = Q inv_LAMBDA Qt
    #  # Q: rotation matrix
    #  # LAMBDA: variances along each main axis
    #  LAMBDA = torch.zeros(B, C, 2, 2)  # (B,C,2,2)
    #  print('           --LAMBDA1:   %1.4f' % (time() - start))
    #  start = time()
    #  LAMBDA[..., 0, 0] = 1 / (lambda_1 * lambda_1)
    #  LAMBDA[..., 1, 1] = 1 / (lambda_2*lambda_2)
    #  # LAMBDA[:, :, 0, 0] = lambda_1**2
    #  # LAMBDA[:, :, 1, 1] = lambda_2**2
    #  print('           --LAMBDA2:   %1.4f' % (time() - start))
    #  start = time()
    #  Q = torch.zeros(B, C, 2, 2)  # (B,C,2,2)
    #  Q[:, :, 0, 0] = torch.cos(thetas)
    #  Q[:, :, 0, 1] = -torch.sin(thetas)
    #  Q[:, :, 1, 0] = torch.sin(thetas)
    #  Q[:, :, 1, 1] = torch.cos(thetas)
    #  print('           --Q:         %1.4f' % (time() - start))
    #  start = time()
    #  # SIGMA = torch.einsum("bcij,bcjk,bckl->bcil", [Q, LAMBDA, Q.transpose(-2, -1)])  # (B,C,2,2)
    #  INV_SIGMA = torch.einsum("bcij,bcjk,bckl->bcil", [Q, LAMBDA, Q.transpose(-2, -1)])  # (B,C,2,2)
    #  print('           --SIGMA:     %1.4f' % (time() - start))
    #  start = time()
    #  # INV_SIGMA = torch.linalg.inv(SIGMA)
    #  INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)
    #  print('           --INV_SIGMA: %1.4f' % (time() - start))

    start = time()
    c = torch.cos(thetas)
    s = torch.sin(thetas)
    cc = c*c
    ss = s*s
    sc = s*c
    # do matrix explicit matrix product and inversion
    LAMBDA1 = 1.0 / (lambda_1 * lambda_1)
    LAMBDA2 = 1.0 / (lambda_2 * lambda_2)
    inv_00 = cc * LAMBDA1 + ss * LAMBDA2
    inv_01 = sc * (LAMBDA1 + LAMBDA2)
    inv_11 = ss * LAMBDA1 + cc * LAMBDA2
    INV_SIGMA = torch.stack([torch.stack([inv_00, inv_01], dim=-1), torch.stack([inv_01, inv_11], dim=-1)], dim=-2)
    INV_SIGMA = INV_SIGMA.view(B,C,1,1,2,2)
    print('           --INV_sigma: %1.4f' % (time() - start))
    # print(torch.linalg.norm(inv_SIGMA.cpu() - INV_SIGMA))


    ## Set expectation position
    #start = time()
    #MU = (ksize//2) * torch.ones(B, C, 2)
    #MU = MU.view(B, C, 1, 1, 2, 1)  # (B,C,1,1,2,1)
    #print('           --MU:        %1.4f' % (time() - start))


    # Calculate Gaussian for every pixel of the kernel
    start = time()
    ZZ = Z
    # ZZ = Z - MU
    ZZ_t = ZZ.transpose(-2, -1)  # (B,C,k,k,1,2)
    kernels = torch.exp(-0.5 * (ZZ_t @ INV_SIGMA @ ZZ).squeeze(-1).squeeze(-1))  # (B,C,k,k)
    print('           --kernels:    %1.4f' % (time() - start))

    # Normalize the kernel and return
    # mask_small = torch.sum(raw_kernels, dim=(-2, -1)) < 1e-2
    # if mask_small.any():
    #     raw_kernels[mask_small].copy_(0)
    #     raw_kernels[mask_small, ksize//2, ksize//2].copy_(1)
    return kernels / (torch.sum(kernels, dim=(-2, -1), keepdim=True) + 1e-8)

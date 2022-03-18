import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from filters import fourier_gradients

import edgetaper
import filters
import blur_estimation


def polyblur(img, n_iter=1, c=0.352, sigma_b=0.768, alpha=2, b=3, masking=False, do_edgetaper=True):
    impred = img
    for n in range(n_iter):
        ## Blur estimation
        kernel = blur_estimation.gaussian_blur_estimation(impred, c=c, sigma_b=sigma_b)
        ## Non-blind deblurring
        impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=b, masking=masking, do_edgetaper=True)
    impred = np.clip(impred, 0.0, 1.0)
    return impred


def inverse_filtering_rank3(img, kernel, alpha=2, b=3, correlate=False, masking=False, do_edgetaper=True):
    if type(img) == np.ndarray:
        return inverse_filtering_rank3_np(img, kernel, alpha, b, correlate, masking, do_edgetaper)
    else:
        return inverse_filtering_rank3_torch(img, kernel, alpha, b, correlate, masking)


def inverse_filtering_rank3_np(img, kernel, alpha=2, b=3, correlate=False, do_masking=False, do_edgetaper=True):
    if img.ndim == 2:
        # img = np.stack([img, img, img], axis=-1)
        # kernel = np.stack([kernel, kernel, kernel], axis=-1)
        img = img[..., None]
        kernel = kernel[..., None]
        flag_gray = True
    else:
        flag_gray = False
        if kernel.ndim == 2:
            kernel = np.stack([kernel, kernel, kernel], axis=-1)
    if correlate:
        kernel = np.rot90(kernel, k=2, axes=(0, 1))
    ## Go to Fourier domain
    if do_edgetaper:
        ks = kernel.shape[0] // 2
        img = np.pad(img, [(ks, ks), (ks, ks), (0, 0)], mode='edge')
        img = [edgetaper.edgetaper(img[..., c], kernel[..., c]) for c in range(img.shape[-1])]  # for better edgehandling
        img = np.stack(img, axis=-1)
    h, w = img.shape[:2]
    Y = np.fft.fft2(img, axes=(0, 1))
    K = [filters.psf2otf(kernel[..., c], (h, w)) for c in range(img.shape[-1])]
    K = np.stack(K, axis=-1)
    ## Compute compensation filter
    C = np.conj(K) / (np.abs(K) + 1e-8)
    a3 = alpha / 2 - b + 2;
    a2 = 3 * b - alpha - 6;
    a1 = 5 - 3 * b + alpha / 2
    Y = C * Y
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    imout = np.real(np.fft.ifft2(X, axes=(0, 1)))
    ## Mask deblurring halos
    if do_masking:
        grad_x, grad_y = fourier_gradients(img)
        gout_x, gout_y = fourier_gradients(imout)
        M = (-grad_x) * gout_x + (-grad_y) * gout_y
        nM = np.sum(np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2, axis=(0, 1))
        z = np.maximum(M / (nM + M), 0)
        imout = z * img + (1 - z) * imout
    if flag_gray:
        imout = np.squeeze(imout, -1)
    if do_edgetaper:
        return imout[ks:-ks, ks:-ks]
    else:
        return imout


def inverse_filtering_rank3_torch(img, kernel, alpha=2, b=3, correlate=False, masking=False):
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))

    ## Go to Fourier domain
    ks = kernel.shape[-1] // 2
    padding = (ks, ks, ks, ks)
    img = F.pad(img, padding, 'replicate')
    h, w = img.shape[-2:]
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = filters.p2o(kernel, (h, w))  # from NxCxhxw to NxCxHxW
    ## Compute compensation filter
    C = torch.conj(K) / (torch.abs(K) + 1e-8)
    a3 = alpha / 2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    Y = C * Y
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    imout = torch.real(torch.fft.ifft2(X, dim=(-2, -1)))
    ## Mask deblurring halos
    if masking:
        grad_x, grad_y = fourier_gradients(img)
        gout_x, gout_y = fourier_gradients(imout)
        M = (-grad_x) * gout_x + (-grad_y) * gout_y
        nM = torch.sum(torch.abs(grad_x) ** 2 + torch.abs(grad_y) ** 2, dim=(-2, -1))
        z = torch.maximum(M / (nM + M), torch.zeros_like(M))
        imout = z * img + (1 - z) * imout
    return imout[..., ks:-ks, ks:-ks]

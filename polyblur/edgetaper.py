import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

from .filters import convolve2d

## Implementation from https://github.com/uschmidt83/fourier-deconvolution-network/blob/master/fdn_predict.py


def edgetaper_alpha(kernel, img_shape):
    z = torch.fft.fft(torch.sum(kernel, -1), img_shape[0]-1)
    z = torch.real(torch.fft.ifft(torch.abs(z)**2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v1 = 1 - z / torch.max(z)

    z = torch.fft.fft(torch.sum(kernel, -2), img_shape[1] - 1)
    z = torch.real(torch.fft.ifft(torch.abs(z) ** 2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v2 = 1 - z / torch.max(z)

    return v1.unsqueeze(-1) * v2.unsqueeze(-2)


def edgetaper(img, kernel, n_tapers=3, method='fft'):
    h, w = img.shape[-2:]
    alpha = edgetaper_alpha(kernel, (h, w))
    _kernel = kernel
    ks = _kernel.shape[-1] // 2
    for i in range(n_tapers):
        blurred = convolve2d(img, kernel, method=method)
        img = alpha * img + (1 - alpha) * blurred
    return img

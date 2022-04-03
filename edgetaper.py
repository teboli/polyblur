import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

from filters import p2o

## Implementation from https://github.com/uschmidt83/fourier-deconvolution-network/blob/master/fdn_predict.py


def edgetaper(img, kernel, n_tapers=3):
    if type(img) == np.ndarray:
        return edgetaper_np(img, kernel, n_tapers)
    else:
        return edgetaper_torch(img, kernel, n_tapers)


def pad_for_kernel_np(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel_np(img, kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2) * [slice(None)]
    return img[r]
 

def edgetaper_alpha_np(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper_np(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha_np(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel_np(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def edgetaper_alpha_torch(kernel, img_shape):
    z = torch.fft.fft(torch.sum(kernel, -1), img_shape[0]-1)
    z = torch.real(torch.fft.ifft(torch.abs(z)**2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v1 = 1 - z / torch.max(z)

    z = torch.fft.fft(torch.sum(kernel, -2), img_shape[1] - 1)
    z = torch.real(torch.fft.ifft(torch.abs(z) ** 2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v2 = 1 - z / torch.max(z)

    return v1.unsqueeze(-1) * v2.unsqueeze(-2)


def edgetaper_torch(img, kernel, n_tapers=3):
    h, w = img.shape[-2:]
    alpha = edgetaper_alpha_torch(kernel, (h, w))
    _kernel = kernel
    ks = _kernel.shape[-1] // 2
    for i in range(n_tapers):
        img_padded = F.pad(img, [ks, ks, ks, ks], mode='circular')
        K = p2o(kernel, img_padded.shape[-2:])
        I = torch.fft.fft2(img_padded)
        blurred = torch.real(torch.fft.ifft2(K * I))[..., ks:-ks, ks:-ks]
        img = alpha * img + (1 - alpha) * blurred
    return img

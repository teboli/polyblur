import torch
import torch.nn.functional as F
import numpy as np
import torch.fft
import scipy.fft


def convolve2d(img, kernel, padding='same', method='direct'):
    """
    A per kernel wrapper for torch.nn.functional.conv2d
    :param img: (B,C,H,W) torch.tensor, the input images
    :param kernel: (B,C,h,w) or (B,1,h,w) torch.tensor, the blur kernels
    :param padding: string, can be 'valid' or 'same' 
    : 
    :return imout: (B,C,H,W) torch.tensor, the filtered images
    """
    if method == 'direct':
        if kernel.shape[1] == img.shape[1]:
            return F.conv2d(img, kernel, groups=img.shape[1], padding=padding)
        else:
            imout = [F.conv2d(img[:,c:c+1], kernel, padding=padding) for c in range(img.shape[1])]
            return torch.cat(imout, dim=1)
    elif method == 'fft':
        ks = kernel.shape[-1]
        X = torch.fft.fft2(F.pad(img, [ks, ks, ks, ks], mode='circular'))
        K = p2o(kernel, X.shape[-2:])
        return torch.real(torch.fft.ifft2(K * X))[..., ks:-ks, ks:-ks]
    else:
        raise('%s is not implemented' % method)


def gaussian_filter(sigma, theta, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
    """"
    Returns a generalized 2D gaussian filter with eigenvalues sigma and angle theta
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1, lambda_2 = sigma
    theta = -theta

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1**2, lambda_2**2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position
    MU = k_size // 2 - shift
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)).astype(np.float32)

    # Normalize the kernel and return
    if np.sum(raw_kernel) < 1e-2:
        kernel = np.zeros_like(raw_kernel)
        kernel[k_size[0]//2, k_size[1]//2] = 1
    else:
        kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def dirac(dims):
    """
    Returns a kernel of size dims with a centered dirac
    """
    kernel = np.zeros(dims)
    hh = dims[0] // 2
    hw = dims[1] // 2
    kernel[hh, hw] = 1
    return kernel


def crop(image, new_size):
    size = image.shape[-2:]
    if size[0] - new_size[0] > 0:
        image = image[..., :new_size[0], :]
    if size[1] - new_size[1] > 0:
        image = image[..., :new_size[1]]
    return image


@torch.jit.script
def fourier_gradients(images):
    """
    Compute the image gradients using Fourier interpolation as in Eq. (21a) and (21b)
        :param images: (B,C,H,W) torch.tensor
        :return grad_x, grad_y: tuple of 2 images of same dimensions as images that
                                are the vertical and horizontal gradients
    """
    ## Find fast size for FFT
    h, w = images.shape[-2:]
    h_fast, w_fast = images.shape[-2:]
    # h_fast = scipy.fft.next_fast_len(h)
    # w_fast = scipy.fft.next_fast_len(w)
    ## compute FT
    U = torch.fft.fft2(images, s=(h_fast, w_fast))
    U = torch.fft.fftshift(U, dim=(-2, -1))
    ## Create the freqs components
    freqh = (torch.arange(0, h_fast, device=images.device) - h_fast // 2)[None, None, :, None] / h_fast
    freqw = (torch.arange(0, w_fast, device=images.device) - w_fast // 2)[None, None, None, :] / w_fast
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-torch.imag(U) + 1j * torch.real(U))
    gxU = torch.fft.ifftshift(gxU, dim=(-2, -1))
    gxu = torch.real(torch.fft.ifft2(gxU))
    # gxu = crop(gxu, (h, w))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    # gyu = crop(gyu, (h, w))
    return gxu, gyu


### From here, taken from https://github.com/cszn/USRNet/blob/master/utils/utils_deblur.py

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf

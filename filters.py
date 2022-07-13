import torch
import numpy as np
import torch.fft
import scipy.fft


def gaussian_filter(sigma, theta, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
    """"
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
    kernel = np.zeros(dims)
    hh = dims[0] // 2
    hw = dims[1] // 2
    kernel[hh, hw] = 1
    return kernel


def crop(image, new_size):
    if type(image) == np.ndarray:
        size = image.shape[:2]
        if size[0] - new_size[0] > 0:
            image = image[size[0] - new_size[0]]
        if size[1] - new_size[1] > 0:
            image = image[:, size[1] - new_size[1]]
    else:
        size = image.shape[-2:]
        if size[0] - new_size[0] > 0:
            image = image[..., size[0] - new_size[0], :]
        if size[1] - new_size[1] > 0:
            image = image[..., size[1] - new_size[1]]
    return image


def fourier_gradients(images):
    if type(images) == np.ndarray:
        return fourier_gradients_np(images)
    else:
        return fourier_gradients_torch(images)


def fourier_gradients_np(images):
    if len(images.shape) == 2:
        images = images[..., None]  # to handle BW images as RGB ones
    ## Find fast size for FFT
    h, w = images.shape[:2]
    h_fast = scipy.fft.next_fast_len(h)
    w_fast = scipy.fft.next_fast_len(w)
    ## compute FT
    U = scipy.fft.fft2(images, s=(h_fast, w_fast), axes=(0, 1))
    U = scipy.fft.fftshift(U, axes=(0, 1))
    ## Create the freqs components
    freqh = (np.arange(0, h_fast, dtype=np.float32) - h_fast // 2)[:, None, None] / h_fast
    freqw = (np.arange(0, w_fast, dtype=np.float32) - w_fast // 2)[None, :, None] / w_fast
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-np.imag(U) + 1j * np.real(U))
    gxU = scipy.fft.ifftshift(gxU, axes=(0, 1))
    gxu = np.real(scipy.fft.ifft2(gxU, axes=(0, 1)))
    gxu = crop(gxu, (h, w))
    gyU = 2 * np.pi * freqh * (-np.imag(U) + 1j * np.real(U))
    gyU = scipy.fft.ifftshift(gyU, axes=(0, 1))
    gyu = np.real(scipy.fft.ifft2(gyU, axes=(0, 1)))
    gyu = crop(gyu, (h, w))
    if images.shape[-1] == 1:
        return np.squeeze(gxu, -1), np.squeeze(gyu, -1)
    else:
        return gxu, gyu


def fourier_gradients_torch(images):
    ## Find fast size for FFT
    h, w = images.shape[-2:]
    h_fast = scipy.fft.next_fast_len(h)
    w_fast = scipy.fft.next_fast_len(w)
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
    gxu = crop(gxu, (h_fast, w_fast))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    gyu = crop(gyu, (h_fast, w_fast))
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


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


def psf2otf(psf, shape=None):
    """
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if type(shape) == type(None):
        shape = psf.shape
    shape = np.array(shape)
    if np.all(psf == 0):
        return np.zeros(shape)
    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))
    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = scipy.fft.fft2(psf, axes=(0, 1))
    return otf

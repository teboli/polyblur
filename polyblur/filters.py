import torch
import torch.nn.functional as F
import numpy as np
import torch.fft
import scipy.fft

from . import utils
# TODO: import separable_gaussian_kernels


#####################################################################
######################## Convolution 2D #############################
#####################################################################

def convolve2d(img, kernel, ksize=25, padding='same', method='direct'):
    # TODO: replace 1d kernels by (sigma,rho,theta)
    """
    A per kernel wrapper for torch.nn.functional.conv2d
    :param img: (B,C,H,W) torch.tensor, the input images
    :param kernel: (B,C,h,w) or 
                   (B,1,h,w) torch.tensor, the 2d blur kernels (valid for both deblurring methods), or 
                   [(B,C,h), (B,C,w)] or 
                   [(B,1,h), (B,1,h)], the separable 1d blur kernels (valid only for spatial deblurring)
    :param padding: string, can be 'valid' or 'same' 
    : 
    :return imout: (B,C,H,W) torch.tensor, the filtered images
    """
    if method == 'direct':
        if type(kernel) == torch.Tensor:  # if we have 2D kernels, do general 2D convolution
            return conv2d_(img, kernel, padding)
        else:                                     # else, do Gaussian-specific 1D separable convolution
            return gaussian_separable_conv2d_(img, kernel, ksize, padding)
    elif method == 'fft':
        assert(type(kernel) == torch.Tensor)  # for FFT, we only use 2D kernels
        X = torch.fft.fft2(utils.pad_with_kernel(img, kernel, mode='circular'))
        K = p2o(kernel, X.shape[-2:])
        return utils.crop_with_kernel( torch.real(torch.fft.ifft2(K * X)), kernel )
    else:
        raise('Convolution method %s is not implemented' % method)


def conv2d_(img, kernel, padding='same'):
    """
    Wrapper for F.conv2d with possibly multi-channels kernels.
    """
    # if the number of kernel matches the number of 
    if kernel.shape[1] == img.shape[1]:  # check how many color channels in the kernel
        return F.conv2d(img, kernel, groups=kernel.shape[1], padding=padding)
    else:
        img = [F.conv2d(img[:,c:c+1], kernel, padding=padding) for c in range(img.shape[1])]
        return torch.cat(img, dim=1)


def gaussian_separable_conv2d_(img, kernel, ksize, padding='same', threshold=1e-4):
    """
    Convolution with separable 1D Gaussian kernels on possibly non-orthogonal axes.
    """
    sigma, rho, theta = kernel
    
    ## First process the orthogonal directions
    mask = (theta % (np.pi / 2)) < threshold  # if theta is 0, 90 or 180 degrees, orthogonal directions
    if mask.any():
        sigma_x = sigma[mask]
        sigma_y = rho[mask]
        img[mask] = gaussian_xy_separable_conv2d_(img[mask], sigma_x, sigma_y, ksize, padding)

    ## Second process the other directions
    mask = (theta % (np.pi / 2)) >= threshold  # else, general case
    if mask.any():
        sigma_u = sigma[mask]
        sigma_v = sigma[mask]
        phi = theta[mask]
        img[mask] = gaussian_xt_sperabla_conv2d_(img[mask], sigma_u, sigma_v, phi, ksize)

    return img
    

def gaussian_xy_separable_conv2d_(img, sigma_x, sigma_y, ksize, padding='same'):
    # Create the 1D kernel along x
    t = torch.arange(-ksize//2 + 1, ksize//2 + 1, device=device).view(1, 1, 1, ksize)
    t = t * t
    kernel = torch.exp( - t / (2 * sigma_x * sigma_x))  # (1, 1, 1, ksize)
    kernel /= kernel.sum()

    # Horizontal filter
    img = F.conv2d(img, kernel, padding='same', groups=img.shape[1])
    img = img.transpose(-1,-2)

    # Create the 1D kernel along y
    kernel = torch.exp( - t / (2 * sigma_y * sigma_y))  # (1, 1, 1, ksize)
    kernel /= kernel.sum()

    # Vertical filter
    img = F.conv2d(img, kernel, padding='same', groups=img.shape[1])
    return img.transpose(-1,-2)


def gaussian_xt_separable_conv2d_(img, sigma, rho, theta, ksize):
    ## Call CUDA code here
    return img



#####################################################################
####################### Bilateral filter ############################
#####################################################################


def bilateral_filter(I, ksize=7, sigma_spatial=5.0, sigma_color=0.1):
    ## precompute the spatial kernel: each entry of gw is a square spatial difference
    t = torch.arange(-ksize//2+1, ksize//2+1, device=I.device)
    xx, yy = torch.meshgrid(t, t, indexing='xy')
    gw = torch.exp(-(xx * xx + yy * yy) / (2 * sigma_spatial * sigma_spatial))  # (ksize, ksize)

    ## Create the padded array for computing the color shifts
    I_padded = utils.pad_with_kernel(I, ksize=ksize)

    ## Filtering
    var2_color = 2 * sigma_color * sigma_color
    return bilateral_filter_loop_(I, I_padded, gw, var2_color, J, W)


def bilateral_filter_loop_(I, I_padded, gw, var2, do_for=True):
    b, c, h, w = I.shape

    if do_for:  # memory-friendly option (Recommanded for larger images)
        J = torch.zeros_like(I)
        W = torch.zeros_like(I)
        for z in range(gw.shape[0] * gw.shape[1]):
            # compute the indices
            x = z % gw.shape[0]
            y = (z-x) // gw.shape[1]
            yy = y + h
            xx = x + w
            # get the shifted image
            I_shifted = I_padded[..., y:yy, x:xx]
            # color weight
            F = I_shifted - I  # (B,C,H,W)
            F = torch.exp(-F * F / var2) 
            # product with spatial weight
            F *= gw[y, x] # (B,C,H,W)
            J += F * I_shifted
            W += F
    else:  # pytorch-friendly option (Faster for smaller images and/or batche sizes)
        # get shifted images
        I_shifted = utils.extract_tiles(I_padded, kernel_size=(h,w), stride=1)  # (B,ksize*ksize,C,H,W)
        F = I_shifted - I.unsqueeze(1)
        F = torch.exp( - F * F / var2)  # (B,ksize*ksize,C,H,W)
        # product with spatial weights
        F *= gw.view(-1, 1, 1, 1)
        J = torch.sum(F * I_shifted, dim=1)  # (B,C,H,W)
        W = torch.sum(F, dim=1)  # (B,C,H,W)
    return J / (W + 1e-8)

    


#####################################################################
###################### Classical filters ############################
#####################################################################


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


def crop(image, new_size):
    size = image.shape[-2:]
    if size[0] - new_size[0] > 0:
        image = image[..., :new_size[0], :]
    if size[1] - new_size[1] > 0:
        image = image[..., :new_size[1]]
    return image


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



#####################################################################
####################### Fourier kernel ##############################
#####################################################################


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

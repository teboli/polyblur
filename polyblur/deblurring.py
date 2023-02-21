import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from . import edgetaper
from . import filters
from . import blur_estimation
from . import domain_transform
from . import utils

from scipy import signal, ndimage, fftpack
from time import time


#####################################################
################## Functional #######################
#####################################################



def polyblur_deblurring(img, n_iter=1, c=0.352, b=0.768, alpha=2, beta=3, sigma_r=0.8, sigma_s=2.0, ker_size=25, q=0.0, n_angles=6,
                        n_interpolated_angles=30, remove_halo=False, edgetaping=False, prefiltering=False, 
                        discard_saturation=False, multichannel_kernel=False, method='fft', verbose=False):
    """
    Meta Functional implementation of Polyblur.
    :param img: (H, W) or (H,W,3) np.array or (B,C,H,W) torch.tensor, the blurry image(s)
    :param n_iter: int, number of polyblur iteration
    :param c: float, slope of std estimation model (C in the main paper)
    :param b: float, intercept of the std estimation model
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :param sigma_r: float, regularization parameter for domain transform (prefiltering)
    :param sigma_s: float, smoothness parameter for domain transform (prefiltering)
    :param ker_size: int, kernel(s) size
    :param remove_halo: bool, using or not halo removal masking
    :param edgetaping: bool, using or not edgetaping border preprocessing for deblurring
    :param prefiltering: bool, using or not smooth and textured images split to prevent noise magnification
    :param discard_saturation: bool, should we discard saturated pixels? (recommanded)
    :param multichannel_kernel: bool, predicting a kernel common to each color channel or a single kernel per color channel
    :param method: string, weither running the convolution in time or fourier domain
    :return: impred: np.array or torch.tensor of same size as img, the restored image(s)
    """
    ## Convert to torch.Tensor from numpy.ndarray
    if type(img) == np.ndarray:
        flag_numpy = True
        img = utils.to_tensor(img).unsqueeze(0)
    else:
        flag_numpy = False

    ## Which format for the kernel: 2D filter (FFT or direct) or tuple of parameters (direct separable)?
    if method == 'direct_separable':
        return_2d_filters = False
    else:
        return_2d_filters = True

    ## Init the variables
    start = time()
    impred = img
    grad_img = filters.fourier_gradients(img)
    thetas = torch.linspace(0, 180, n_angles+1, device=img.device).unsqueeze(0).long()   # (1,n)
    interpolated_thetas = torch.arange(0, 180, 180 / n_interpolated_angles, device=img.device).unsqueeze(0).long()  # (1,N)
    if verbose:
        print('-- init tensors:      %1.5f' % (time() - start))

    ## Main loop
    for n in range(n_iter):
        ## Blur estimation
        start = time()
        kernel = blur_estimation.gaussian_blur_estimation(impred, c=c, b=b, q=q, discard_saturation=discard_saturation, 
                                                          ker_size=ker_size, multichannel=multichannel_kernel, 
                                                          thetas=thetas, interpolated_thetas=interpolated_thetas,
                                                          return_2d_filters=return_2d_filters)
        if verbose:
            print('-- blur estimation %d: %1.5f' % (n+1, time() - start))

        ## Non-blind deblurring
        start =time()
        if prefiltering:
            impred, impred_noise = edge_aware_filtering(impred, sigma_s, sigma_r)
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, remove_halo=remove_halo,
                                             do_edgetaper=edgetaping, grad_img=grad_img, method=method)
            impred = impred + impred_noise
        else:
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, remove_halo=remove_halo,
                                             do_edgetaper=edgetaping, grad_img=grad_img, method=method)
        impred = impred.clip(0.0, 1.0)
        if verbose:
            print('-- deblurring %d:      %1.5f' % (n+1, time() - start))

    ## Go back to numpy if needs be
    if flag_numpy:
        return utils.to_array(impred.cpu())
    else:
        return impred


def edge_aware_filtering(img, sigma_s, sigma_r):
    """
    Separate img into smooth and detail (noise) components using the the edge aware smoothing. (Alg.4 and Alg.6)
    :param img: (B,C,H,W) torch.tensor, the input image(s)
    :param sigma_r: float, regularization parameter for domain transform
    :param sigma_s: float, smoothness parameter for domain transform
    :return: img_smoothed, img_noise: torch.tensors of same size as img, the smooth and noise components of img
    """
    # img_smoothed = domain_transform.recursive_filter(img, sigma_r=sigma_r, sigma_s=sigma_s, num_iterations=1)
    img_smoothed = filters.bilateral_filter(img)
    img_noise = img - img_smoothed
    return img_smoothed, img_noise


def compute_polynomial(img, kernel, alpha, b, method='fft', not_symmetric=False):
    if method == 'fft':
        return compute_polynomial_fft(img, kernel, alpha, b, not_symmetric)
    elif method == 'direct' or method == 'direct_separable':
        return compute_polynomial_direct(img, kernel, alpha, b, not_symmetric)
    else:
        Exception('%s not implemented' % method)


def compute_polynomial_direct(img, kernel, alpha, b, not_symmetric=False):
    """
    Implements in the time domain the polynomial deconvolution filter (Deconvolution Alg.4) 
    using the polynomial approximation of Eq. (27)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (B,C,h,w) or (B,1,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    a3 = alpha/2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    imout = a3 * img
    imout = filters.convolve2d(imout, kernel) + a2 * img
    imout = filters.convolve2d(imout, kernel) + a1 * img
    return filters.convolve2d(imout, kernel) + b * img


def compute_polynomial_fft(img, kernel, alpha, b, not_symmetric=False):
    """
    Implements in the fourier domain the polynomial deconvolution filter (Deconvolution Alg.4) 
    using the polynomial approximation of Eq. (27)
    :param Y: (B,C,H,W) torch.tensor, the blurry image(s)
    :param K: (B,C,h,w)  or (B,1,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    ## Go to Fourier domain
    h, w = img.shape[-2:]
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = filters.p2o(kernel, (h, w))  # from NxCxhxw to NxCxHxW
    ## If the filter is not symmetric, shift the image
    if not_symmetric:
        C = torch.conj(K) / (torch.abs(K) + 1e-8)
        Y = C * Y
    ## C implements the pure phase filter described in the Polyblur article 
    ## needed to deblur non symmetic kernels. For Gaussian kernels (symmetric) it has no effect.
    a3 = alpha / 2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    ## Go back to temporal domain
    return torch.real(torch.fft.ifft2(X, dim=(-2, -1)))


# @torch.jit.script
def grad_prod_(grad_x, grad_y, gout_x, gout_y):
    return (- grad_x * gout_x) +  (- grad_y * grad_y)


# @torch.jit.script
def grad_square_(grad_x, grad_y):
    return grad_x * grad_x + grad_y * grad_y


# @torch.jit.script
def grad_div_and_clip_(M, nM):
    return torch.clamp(M / (nM + M), min=0)


# @torch.jit.script
def grad_convex_sum_(img, imout, z):
    # Equivalent to z * img + (1-z) * imout
    return imout + z * (img - imout)


def halo_masking(img, imout, grad_img=None):
    """
    Halo removal processing. Detects gradient inversions between input and deblurred image replaces them in the output (Alg.5)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param imout: (B,C,H,W) torch.tensor, the deblurred image(s)
    :return torch.tensor of same size as img, the halo corrected image(s)
    """
    if grad_img is None:
        grad_x, grad_y = filters.fourier_gradients(img)
    else:
        grad_x, grad_y = grad_img
    gout_x, gout_y = filters.fourier_gradients(imout)
    M = grad_prod_(grad_x, grad_y, gout_x, gout_y)
    nM = torch.sum(grad_square_(grad_x, grad_y), dim=(-2, -1), keepdim=True)
    z = grad_div_and_clip_(M, nM)
    return grad_convex_sum_(img, imout, z)


def inverse_filtering_rank3(img, kernel, alpha=2, b=4, correlate=False, remove_halo=False, 
                            do_edgetaper=False, grad_img=None, method='direct'):
    """
    Deconvolution with approximate inverse filter parameterized by alpha and beta. (Deconvolution Alg.4, EdgeAwareFiltering is in Alg.1)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (B,C,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter
    :param beta: float, high frequencies parameter
    :param correlate: bool, deconvolving with a correlation or not
    :param masking: bool, using or not halo removal masking
    :param do_edgetaper bool, using or not edgetaping border preprocessing for deblurring
    :param method string, weither running the convolution in time or fourier domain
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
    ## Pad (with edgetaping optionally)
    img = utils.pad_with_kernel(img, kernel)
    if do_edgetaper:
        img = edgetaper.edgetaper(img, kernel, method=method)  # for better edge handling
    ## Deblurring
    imout = compute_polynomial(img, kernel, alpha, b, method=method)
    ## Crop
    imout = utils.crop_with_kernel(imout, kernel)
    ## Mask deblurring halos
    if remove_halo:
        img = utils.crop_with_kernel(img, kernel)
        imout = halo_masking(img, imout, grad_img)
    return torch.clamp(imout, 0.0, 1.0)





#####################################################
##################  nn.Module  ######################
#####################################################


class PolyblurDeblurring(nn.Module):
    def __init__(self, patch_decomposition=False, patch_size=400, patch_overlap=0.25, batch_size=1):
        """
        A nn.Module wrapper of deblurring.polyblur with inner patch decomposition if processing large images
        where the blur may vary.
        :param patch_decomposition: bool, decomposing the image in patches or not
        :param patch_size: int, size of a square patch
        :param patch_overlap: float, percentage of patch overlapping
        :param batch_size: int, number of patches processed at once in polyblur (for memory constraints)
        """
        super(PolyblurDeblurring, self).__init__()
        self.batch_size = batch_size
        self.patch_decomposition = patch_decomposition
        self.patch_size = (patch_size, patch_size)
        self.patch_overlap = patch_overlap

    def forward(self, images, n_iter=1, c=0.352, b=0.468, alpha=2, beta=4, sigma_s=2, ker_size=25, sigma_r=0.4,
                q=0.0, n_angles=6, n_interpolated_angles=30, remove_halo=False, edgetaping=False, prefiltering=False, 
                discard_saturation=False, multichannel_kernel=False, method='fft', device=None):
        if self.patch_decomposition:
            patch_size = self.patch_size

            ## Make sure dimensions are even
            h, w = images.shape[-2:]
            if h % 2 == 1:
                images = images[..., :-1, :]
                h -= 1
            if w % 2 == 1:
                images = images[..., :, :-1]
                w -= 1

            ## Pad the image if needed
            step_h = int(patch_size[0] * (1 - self.patch_overlap))
            step_w = int(patch_size[1] * (1 - self.patch_overlap))
            new_h = int(np.ceil((h - patch_size[0]) / step_h) * step_h) + patch_size[0]
            new_w = int(np.ceil((w - patch_size[1]) / step_w) * step_w) + patch_size[1]

            images_padded = self.pad_with_new_size(images, (new_h, new_w), mode='replicate')

            if handling_saturation:
                masks_padded = images_padded > 0.99  # Assuming pixel values or in [0,1]

            ## Get indice
            # s of the top-left corners of the patches
            I_coords = np.arange(0, new_h - patch_size[0] + 1, step_h)
            J_coords = np.arange(0, new_w - patch_size[1] + 1, step_w)
            IJ_coords = np.meshgrid(I_coords, J_coords, indexing='ij')
            IJ_coords = np.stack(IJ_coords).reshape(2, -1).T  # (N,2)
            n_blocks = len(I_coords) * len(J_coords)

            ## Create the arrays for outputing results
            ph = patch_size[0]
            pw = patch_size[1]
            window = self.build_window(patch_size, window_type='kaiser').unsqueeze(0).unsqueeze(0).to(images.device)  # (1,1,h,w)

            images_restored = torch.zeros_like(images_padded)  # (B,C,H,W)
            window_sum = torch.zeros(1, 1, images_padded.shape[-2], images_padded.shape[-1], device=images.device)  # (1,1,H,W)

            ### End of get patch coordinates

            for m in range(0, n_blocks, self.batch_size):
                ## Extract the patches
                IJ_coords_batch = IJ_coords[m:m + self.batch_size]  # (B,2)
                patches = [images_padded[..., i0:i0 + ph, j0:j0 + pw] for (i0, j0) in IJ_coords_batch]
                patches = torch.cat(patches, dim=0)  # (B*batch_size,C,h,w) : [0,0,0,0, 1,1,1,1, 2,2,2,2,....]
                if handling_saturation:
                    masks = [masks_padded[..., i0:i0 + ph, j0:j0 + pw] for (i0, j0) in IJ_coords_batch]
                    masks = torch.cat(masks, dim=0)
                else:
                    masks = None

                ## Deblurring
                if device is not None:
                    patches = patches.to(device)
                patches_restored = polyblur_deblurring(patches, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, ker_size=ker_size,
                                                       sigma_s=sigma_s, sigma_r=sigma_r, remove_halo=remove_halo, edgetaping=edgetaping,
                                                       prefiltering=prefiltering, discard_saturation=discard_saturation,
                                                       multichannel_kernel=multichannel_kernel, method=method, q=q, n_angles=n_angles,
                                                       n_interpolated_angles=n_interpolated_angles)
                if device is not None:
                    patches_restored = patches_restored.cpu()

                ## Replace the patches
                for n in range(IJ_coords_batch.shape[0]):
                    i0, j0 = IJ_coords_batch[n]
                    images_restored[..., i0:i0 + ph, j0:j0 + pw] += patches_restored[n::self.batch_size] * window
                    window_sum[..., i0:i0 + ph, j0:j0 + pw] += window

            images_restored = images_restored / (window_sum + 1e-8)
            images_restored = images_restored.clamp(0.0, 1.0)
            images_restored = self.crop_with_old_size(images_restored, (h, w))
        else:
            images_restored = polyblur_deblurring(images, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, ker_size=ker_size,
                                                  sigma_s=sigma_s, sigma_r=sigma_r, remove_halo=remove_halo, edgetaping=edgetaping,
                                                  prefiltering=prefiltering, discard_saturation=discard_saturation,
                                                  multichannel_kernel=multichannel_kernel, method=method, q=q, n_angles=n_angles,
                                                  n_interpolated_angles=n_interpolated_angles)
        return images_restored

    def build_window(self, image_size, window_type):
        H, W = image_size
        if window_type == 'kaiser':
            window_i = torch.kaiser_window(H, beta=5, periodic=True)
            window_j = torch.kaiser_window(W, beta=5, periodic=True)
        elif window_type == 'hann':
            window_i = torch.hann_window(H, periodic=True)
            window_j = torch.hann_window(W, periodic=True)
        elif window_type == 'hamming':
            window_i = torch.hamming_window(H, periodic=True)
            window_j = torch.hamming_window(W, periodic=True)
        elif window_type == 'bartlett':
            window_i = torch.bartlett_window(H, periodic=True)
            window_j = torch.bartlett_window(W, periodic=True)
        else:
            Exception('Window not implemented')

        return window_i.unsqueeze(-1) * window_j.unsqueeze(0)

    def pad_with_new_size(self, img, new_size, mode='constant'):
        h, w = img.shape[-2:]
        new_h, new_w = new_size
        pad_left = int(np.floor((new_w - w)/2))
        pad_right = int(np.ceil((new_w - w) / 2))
        pad_top = int(np.floor((new_h - h)/2))
        pad_bottom = int(np.ceil((new_h - h) / 2))
        padding = [pad_left, pad_right, pad_top, pad_bottom]
        img = F.pad(img, padding, mode=mode)
        return img

    def crop_with_old_size(self, img, old_size):
        h, w, = img.shape[-2:]
        old_h, old_w = old_size
        crop_left = int(np.floor((w - old_w)/2))
        if crop_left > 0:
            img = img[..., :, crop_left:]
        crop_right = int(np.ceil((w - old_w) / 2))
        if crop_right > 0:
            img = img[..., :, :-crop_right]
        crop_top = int(np.floor((h - old_h) / 2))
        if crop_top > 0:
            img = img[..., crop_top:, :]
        crop_bottom = int(np.ceil((h - old_h) / 2))
        if crop_bottom > 0:
            img = img[..., :-crop_bottom, :]
        return img

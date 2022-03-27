import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from filters import fourier_gradients

import edgetaper
import filters
import blur_estimation
import domain_transform


#####################################################
#############  Numpy/Pytorch routine  ###############
#####################################################


def polyblur(img, n_iter=1, c=0.352, b=0.768, alpha=2, beta=3, sigma_r=0.8, sigma_s=2, ker_size=25, masking=False,
             edgetaping=False, prefiltering=False, saturation_mask=None, multichannel_kernel=False):
    """
    Functional implementation of Polyblur.
    :param img: (H,W) or (H,W,3) np.array or (B,C,H,W) torch.tensor, the blurry image(s)
    :param n_iter: int, number of polyblur iteration
    :param c: float, slope of std estimation model (C in the main paper)
    :param b: float, intercept of the std estimation model
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :param sigma_r: float, regularization parameter for domain transform (prefiltering)
    :param sigma_s: float, smoothness parameter for domain transform (prefiltering)
    :param ker_size: int, kernel(s) size
    :param masking: bool, using or not halo removal masking
    :param edgetaping: bool, using or not edgetaping border preprocessing for deblurring
    :param prefiltering: bool, using or not smooth and textured images split to prevent noise magnification
    :param saturation_mask: bool, np.array or torch.tensor of same size as img, mask to exclude pixel values in the gradients, e.g., for saturation
    :param multichannel_kernel: bool, predicting a kernel common to each color channel or a single kernel per color channel
    :return: impred: np.array or torch.tensor of same size as img, the restored image(s)
    """
    impred = img
    ## Main loop
    for n in range(n_iter):
        ## Blur estimation
        kernel = blur_estimation.gaussian_blur_estimation(impred, c=c, b=b, mask=saturation_mask, ker_size=ker_size,
                                                          multichannel=multichannel_kernel)
        ## Non-blind deblurring
        if prefiltering:
            impred, impred_noise = edge_aware_filtering(impred, sigma_s, sigma_r)
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, beta=beta, masking=masking,
                                             do_edgetaper=edgetaping)
            impred = impred + impred_noise
        else:
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, beta=beta, masking=masking,
                                             do_edgetaper=edgetaping)
        impred = np.clip(impred, 0.0, 1.0)
    return impred


def edge_aware_filtering(img, sigma_s, sigma_r):
        img_smoothed = domain_transform.recursive_filter(img, sigma_r=sigma_r, sigma_s=sigma_s)
        img_noise = img - img_smoothed
        return img_smoothed, img_noise


def inverse_filtering_rank3(img, kernel, alpha=2, beta=3, correlate=False, masking=False, do_edgetaper=True):
    """
    Deconvolution with approximate inverse filter parameterized by alpha and beta.
    :param img: (H,W) or (H,W,3) np.array or (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (h,w) or (h,w,3) np.array or (B,C,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter
    :param beta: float, high frequencies parameter
    :param correlate: bool, deconvolving with a correlation or not
    :param masking: bool, using or not halo removal masking
    :param do_edgetaper bool, using or not edgetaping border preprocessing for deblurring
    :return np.array or torch.tensor of same size as img, the deblurred image(s)
    """
    if type(img) == np.ndarray:
        return inverse_filtering_rank3_np(img, kernel, alpha, beta, correlate, masking, do_edgetaper)
    else:
        return inverse_filtering_rank3_torch(img, kernel, alpha, beta, correlate, masking, do_edgetaper)


def compute_polynomial(Y, K, C, alpha, b):
    a3 = alpha / 2 - b + 2;
    a2 = 3 * b - alpha - 6;
    a1 = 5 - 3 * b + alpha / 2
    Y = C * Y
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    return X


def halo_masking(img, imout):
    grad_x, grad_y = fourier_gradients(img)
    gout_x, gout_y = fourier_gradients(imout)
    M = (-grad_x) * gout_x + (-grad_y) * gout_y
    if type(img) == np.ndarray:
        nM = np.sum(grad_x ** 2 + grad_y ** 2, axis=(0, 1))
        z = np.maximum(M / (nM + M), 0)
    else:
        nM = torch.sum(grad_x ** 2 + grad_y ** 2, dim=(-2, -1), keepdim=True)
        z = torch.maximum(M / (nM + M), torch.zeros_like(M))
    imout = z * img + (1 - z) * imout
    return imout


def inverse_filtering_rank3_np(img, kernel, alpha=2, b=3, correlate=False, do_masking=False, do_edgetaper=True):
    if img.ndim == 2:
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
    C = np.conj(K) / (np.abs(K) + 1e-8)
    ## Compute compensation filter
    X = compute_polynomial(Y, K, C, alpha, b)
    imout = np.real(np.fft.ifft2(X, axes=(0, 1)))
    ## Mask deblurring halos
    if do_masking:
        imout = halo_masking(img, imout)
    if flag_gray:
        imout = np.squeeze(imout, -1)
    if do_edgetaper:
        return imout[ks:-ks, ks:-ks]
    else:
        return imout


def inverse_filtering_rank3_torch(img, kernel, alpha=2, b=3, correlate=False, masking=False, do_edgetaper=True):
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
    ## Go to Fourier domain
    if do_edgetaper:
        ks = kernel.shape[-1] // 2
        img = F.pad(img, (ks, ks, ks, ks), mode='replicate')
        img = edgetaper.edgetaper(img, kernel)  # for better edge handling
    # ks = kernel.shape[-1] // 2
    # padding = (ks, ks, ks, ks)
    # img = F.pad(img, padding, 'replicate')
    h, w = img.shape[-2:]
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = filters.p2o(kernel, (h, w))  # from NxCxhxw to NxCxHxW
    C = torch.conj(K) / (torch.abs(K) + 1e-8)
    ## Compute compensation filter
    X = compute_polynomial(Y, K, C, alpha, b)
    imout = torch.real(torch.fft.ifft2(X, dim=(-2, -1)))
    ## Mask deblurring halos
    if masking:
        imout = halo_masking(img, imout)
    if do_edgetaper:
        return imout[..., ks:-ks, ks:-ks]
    else:
        return imout




#####################################################
###############  Pytorch nn.Module  #################
#####################################################


class Polyblur(nn.Module):
    def __init__(self, patch_decomposition=False, patch_size=400, patch_overlap=0.25, batch_size=1):
        """
        A nn.Module wrapper of deblurring.polyblur with inner patch decomposition if processing large images
        where the blur may vary.
        :param patch_decomposition: bool, decomposing the image in patches or not
        :param patch_size: int, size of a square patch
        :param patch_overlap: float, percentage of patch overlapping
        :param batch_size: int, number of patches processed at once in polyblur (for memory constraints)
        """
        super(Polyblur, self).__init__()
        self.batch_size = batch_size
        self.patch_decomposition = patch_decomposition
        self.patch_size = (patch_size, patch_size)
        self.patch_overlap = patch_overlap

    def forward(self, images, n_iter=1, c=0.352, b=0.468, alpha=2, beta=4, sigma_s=2, ker_size=25, sigma_r=0.4,
                masking=False, edgetaping=False, prefiltering=False, handling_saturation=False,
                multichannel_kernel=False):
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

            images_restored = torch.zeros_like(images_padded)  # (C,H,W)
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
                patches_restored = polyblur(patches, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, ker_size=ker_size,
                                            sigma_s=sigma_s, sigma_r=sigma_r, masking=masking, edgetaping=edgetaping,
                                            prefiltering=prefiltering, saturation_mask=masks,
                                            multichannel_kernel=multichannel_kernel)

                ## Replace the patches
                for n in range(IJ_coords_batch.shape[0]):
                    i0, j0 = IJ_coords_batch[n]
                    images_restored[..., i0:i0 + ph, j0:j0 + pw] += patches_restored[n::self.batch_size] * window
                    window_sum[..., i0:i0 + ph, j0:j0 + pw] += window

            images_restored = images_restored / (window_sum + 1e-8)
            images_restored = images_restored.clamp(0.0, 1.0)
            images_restored = self.crop_with_old_size(images_restored, (h, w))
        else:
            if handling_saturation:
                masks = images > 0.99
            else:
                masks = None
            images_restored = polyblur(images, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, ker_size=ker_size,
                                       sigma_s=sigma_s, sigma_r=sigma_r, masking=masking, edgetaping=edgetaping,
                                       prefiltering=prefiltering, saturation_mask=masks,
                                       multichannel_kernel=multichannel_kernel)
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

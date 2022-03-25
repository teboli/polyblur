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


def polyblur(img, n_iter=1, c=0.352, b=0.768, alpha=2, beta=3, sigma_r=0.8, sigma_s=2, masking=False, edgetaping=False,
             prefiltering=False, handling_saturation=False):
    impred = img
    ## Main loop
    for n in range(n_iter):
        ## Blur estimation
        kernel = blur_estimation.gaussian_blur_estimation(impred, c=c, sigma_b=b,
                                                          handling_saturation=handling_saturation)
        ## Non-blind deblurring
        if prefiltering:
            impred, impred_noise = edge_aware_filtering(impred, sigma_s, sigma_r)
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, masking=masking,
                                             do_edgetaper=edgetaping)
            impred = impred + impred_noise
        else:
            impred = inverse_filtering_rank3(impred, kernel, alpha=alpha, b=beta, masking=masking,
                                             do_edgetaper=edgetaping)
        impred = np.clip(impred, 0.0, 1.0)
    return impred


def edge_aware_filtering(img, sigma_s, sigma_r):
        img_smoothed = domain_transform.recursive_filter(img, sigma_r=sigma_r, sigma_s=sigma_s)
        img_noise = img - img_smoothed
        return img_smoothed, img_noise


def inverse_filtering_rank3(img, kernel, alpha=2, b=3, correlate=False, masking=False, do_edgetaper=True):
    if type(img) == np.ndarray:
        return inverse_filtering_rank3_np(img, kernel, alpha, b, correlate, masking, do_edgetaper)
    else:
        return inverse_filtering_rank3_torch(img, kernel, alpha, b, correlate, masking)


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
        nM = torch.sum(grad_x ** 2 + grad_y ** 2, dim=(-2, -1))
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
        ks = kernel.shape[0] // 2
        img = F.pad(img, (ks, ks, ks, ks), mode='circular')
        img = edgetaper.edgetaper(img, kernel)  # for better edge handling
    ks = kernel.shape[-1] // 2
    padding = (ks, ks, ks, ks)
    img = F.pad(img, padding, 'replicate')
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
        return imout[ks:-ks, ks:-ks]
    else:
        return imout




#####################################################
###############  Pytorch nn.Module  #################
#####################################################


class Polyblur(nn.Module):
    def __init__(self, patch_decomposition=False, patch_overlap=0.25):
        super(Polyblur, self).__init__()

        self.patch_decomposition = patch_decomposition
        self.patch_overlap = patch_overlap

    def forward(self, blurry, n_iter=1, c=0.352, b=0.468, alpha=2, beta=4, masking=False, edgetaping=False,
                prefiltering=False):
        if self.patch_decomposition:
            ##### get patch coordinates ####

            ## Make sure dimensions are even
            h, w = image.shape[:2]
            if h % 2 == 1:
                image = image[:-1, :]
                h -= 1
            if w % 2 == 1:
                image = image[:, :-1]
                w -= 1

            ## Pad the image if needed
            new_h = int(np.ceil(h / patch_size[0]) * patch_size[0])
            new_w = int(np.ceil(w / patch_size[1]) * patch_size[1])
            img_padded = pad_with_new_size(image, (new_h, new_w), mode='edge')

            ## Get indices of the top-left corners of the patches
            I_coords = np.arange(0, new_h - patch_size[0] + 1, int(patch_size[0] * (1 - overlap_percentage)))
            J_coords = np.arange(0, new_w - patch_size[1] + 1, int(patch_size[1] * (1 - overlap_percentage)))
            IJ_coords = np.meshgrid(I_coords, J_coords, indexing='ij')
            IJ_coords = np.stack(IJ_coords).reshape(2, -1).T  # (N,2)
            n_blocks = len(I_coords) * len(J_coords)

            ## Create the arrays for outputing results
            ph = patch_size[0]
            pw = patch_size[1]
            window = build_window(patch_size, window_type='kaiser').unsqueeze(0).to(device)  # (1,h,w)

            img_padded = utils.to_tensor(img_padded).to(device)  # (C,H,W)
            img_restored = torch.zeros_like(img_padded)  # (C,H,W)
            window_sum = torch.zeros(1, img_padded.shape[1], img_padded.shape[2], device=img_padded.device)  # (1,H,W)

            ### End of get patch coordinates

            for b in range(0, n_blocks, batch_size):
                ## Extract the patches

                ##### exctract patches #####

                IJ_coords_batch = IJ_coords[b:b + batch_size]  # (B,2)
                patches = [img_padded[:, i0:i0 + ph, j0:j0 + pw] for (i0, j0) in IJ_coords_batch]
                patches = torch.stack(patches, dim=0)  # (B,C,h,w)

                ##### End of extract patches ####

                ## Deblurring
                patches_restored = polyblur(patches_blurry, n_iter, c, b, alpha, beta, masking, edgetaping, prefiltering)

                ## Replace the patches
                for n in range(IJ_coords_batch.shape[0]):
                    i0, j0 = IJ_coords_batch[n]
                    img_restored[:, i0:i0 + ph, j0:j0 + pw] += patch_red_restored[n] * window
                    img_restored[1:2, i0:i0 + ph, j0:j0 + pw] += patches[n, 1:2] * window
                    img_restored[2:3, i0:i0 + ph, j0:j0 + pw] += patch_blue_restored[n] * window

                    window_sum[:, i0:i0 + ph, j0:j0 + pw] += window

            img_restored = img_restored / (window_sum + 1e-8)
            img_restored = img_restored.clamp(0.0, 1.0)
            print('It took %d seconds' % (time.time() - start))
            img_restored = utils.to_array(img_restored.cpu())  # (H,W,C)
            img_restored = crop_with_old_size(img_restored, (h, w))
        else:
            restored = polyblur(blurry)
        return restored

    def get_patch_coordinates(self):
        return

    def extract_patches(self):
        return

    def put_back_patches(self):
        return
